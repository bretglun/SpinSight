import re
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import scipy
import svgpathtools
import toml
from tqdm import tqdm

from spinsight import constants, recon


def polygon_area(coords):
    return np.sum((coords[0]-np.roll(coords[0], 1)) * (coords[1]+np.roll(coords[1], 1))) / 2


def get_vertices(continuous_path):
    if not continuous_path.isclosed():
        raise ValueError('All paths in SVG file must be closed')
    vertices = []
    for segment in continuous_path:
        if isinstance(segment, svgpathtools.path.Line):
            vertices.append((segment[0].imag, segment[0].real))
        else:
            warnings.warn(f'Only SVG line segments are supported, not {type(segment)}')
            return None
    if vertices:
        return np.array(vertices).T
    return None


def parse_style_string(str):
    return dict(attr.strip().split(':') for attr in str.strip(r' ;\t\n').split(';'))


def get_hexcolor(attrib, styles):
    if 'style' in attrib:
        style = parse_style_string(attrib['style'])
    elif 'class' in attrib and attrib['class'] in styles:
        style = styles[attrib['class']]
    else:
        return None
    if 'fill' in style:
        return style['fill']
    return None


def get_styles(file):
    styles = {}
    style_element = ET.parse(file).getroot().find('{http://www.w3.org/2000/svg}style')
    if style_element is not None:
        for class_name, style_string in re.findall(r'\.(\w+)\s*\{([^}]*)\}', style_element.text):
            styles[class_name] = parse_style_string(style_string)
    return styles


def transform_coordinates(coords, transform):
    return np.dot(transform, np.vstack([coords, np.ones((1, coords.shape[1]))]))[:2, :]


def kspace_for_polygon(poly, k):
    # analytical 2D Fourier transform of polygon (see https://cvil.ucsd.edu/wp-content/uploads/2016/09/Realistic-analytical-polyhedral-MRI-phantoms.pdf)
    r = poly['vertices'] # position vectors of vertices Ve
    Lv = np.roll(r, -1, axis=1) - r # edge vectors
    L = np.linalg.norm(Lv, axis=0) # edge lengths
    t = Lv/L # edge unit vectors
    n = np.array([-t[1,:], t[0,:]]) # normals to tangents (pointing out from polygon)
    rc = r + Lv / 2 # position vector for center of edge

    ksp = np.sum(L * np.dot(k, n) * np.sinc(np.dot(k, Lv)) * np.exp(-2j*np.pi * np.dot(k, rc)), axis=-1)
    
    kcenter = np.all(k==0, axis=-1)
    ksp[kcenter] = polygon_area(r)
    notkcenter = np.logical_not(kcenter)
    ksp[notkcenter] *= 1j / (2 * np.pi * np.linalg.norm(k[notkcenter], axis=-1)**2)
    return ksp


def kspace_for_ellipse(ellipse, k):
    # scaled and rotated radius vector:
    r = np.linalg.norm(k @ recon.rotation_matrix(np.radians(ellipse['angle'])) @ np.diag(ellipse['radius']), axis=-1)
    ksp = np.empty(k.shape[:-1], dtype=complex)
    ksp[r!=0] = scipy.special.j1(2 * np.pi * r[r!=0]) / r[r!=0] # Bessel function of first kind, first order
    ksp[r==0] = np.pi
    ksp *= np.prod(ellipse['radius']) # scale intensity
    ksp *= np.exp(-2j*np.pi * np.dot(k, ellipse['pos'])) # translate
    if ellipse['negative']:
        return -ksp
    return ksp


def offset_FOV(kspace, k_grid, offset):
    return kspace * np.exp(2j * np.pi * np.dot(k_grid, offset))


KSPACE_SHAPE_FUNC = {
    "polygon": kspace_for_polygon,
    "ellipse": kspace_for_ellipse,
}


def kspace_for_shape(shape, k_axes, k_grid, axes_support, axes_center):
    shape_support, shape_center = get_support_and_center([shape])
    matrix = np.array([len(axis) for axis in k_axes])
    sparse_matrix = np.ceil(matrix * shape_support / axes_support).astype(int)
    sparse_support = axes_support * sparse_matrix / matrix
    sparse_k_axes = get_k_axes(sparse_matrix, sparse_support)
    sparse_k_grid = get_k_grid(sparse_k_axes)
    kspace = KSPACE_SHAPE_FUNC[shape['type']](shape, sparse_k_grid)
    kspace = offset_FOV(kspace, sparse_k_grid, shape_center)
    kernels = recon.sinc_kernels(k_axes, sparse_k_axes, sparse_support)
    kspace = recon.sinc_interpolate(kspace, kernels)
    return offset_FOV(kspace, k_grid, axes_center - shape_center)


def colormap_shapes(shapes, colormap_file):
    if not colormap_file.is_file():
        raise FileNotFoundError(f'Expected colormap file at {colormap_file}')
    with open(colormap_file, 'r') as f:
        colormap = toml.load(f)
    for hexcolor in list(shapes.keys()):
        colored_shapes = shapes.pop(hexcolor)
        if hexcolor in colormap:
            tissue = colormap[hexcolor]
            if tissue in constants.TISSUES:
                shapes[tissue] = colored_shapes
            else:
                warnings.warn(f'Tissue "{tissue}" from colormap {colormap_file} not defined in constants.py')
        else:
            warnings.warn(f'Tissue color {hexcolor} in phantom not defined in colormap {colormap_file}')
    return shapes


# reads SVG file and returns polygon lists
def load_shapes_svg(file):
    styles = get_styles(file)
    paths, attributes = svgpathtools.svg2paths(file)
    shapes = {}
    for path, attrib in zip(paths, attributes):
        hexcolor = get_hexcolor(attrib, styles)
        if hexcolor not in shapes:
            shapes[hexcolor] = []
        transform = svgpathtools.parser.parse_transform(attrib.get('transform', ''))
        subpaths = path.continuous_subpaths()
        polys = []
        for subpath in subpaths:
            vertices = get_vertices(subpath)
            if vertices is not None:
                polys.append({'type': 'polygon', 'vertices': transform_coordinates(vertices, transform)})
        if sum([polygon_area(poly['vertices']) for poly in polys]) < 0:
            # invert polygons to make total area positive
            for poly in polys:
                poly['vertices'] = np.flip(poly['vertices'], axis=1)
        shapes[hexcolor] += polys
    return colormap_shapes(shapes, file.with_suffix('.toml'))


def load_shapes_toml(file):
    with open(file, 'r') as f:
        shapes = toml.load(f)
    shapes = {k.replace('_', ' '): v for k, v in shapes.items()}
    for ellipse in (shape for lst in shapes.values() for shape in lst if shape['type']=='ellipse'):
        for key in ['pos', 'radius']:
            ellipse[key].reverse()
    return shapes


def load_shapes(file):
    match file.suffix:
        case '.svg':
            return load_shapes_svg(file)
        case '.toml':
            return load_shapes_toml(file)


def get_bounds(shape):
    match shape['type']:
        case 'polygon':
            return np.min(shape['vertices'], axis=1), np.max(shape['vertices'], axis=1)
        case 'ellipse':
            pos = np.array(shape['pos'])
            rotmat = recon.rotation_matrix(np.radians(shape['angle']))
            half_extent = np.abs(rotmat) @ np.array(shape['radius'])
            return pos - half_extent, pos + half_extent
        case _:
            raise ValueError(f'Unknown shape {shape["type"]}')


def get_support_and_center(shapes):
    minima, maxima = zip(*(get_bounds(shape) for shape in shapes))
    minimum = np.min(minima, axis=0)
    maximum = np.max(maxima, axis=0)
    support = maximum - minimum
    center = (maximum + minimum) / 2
    return support, center


def get_k_axes(matrix, support):
    return [recon.get_k_axis(matrix[dim], support[dim]/matrix[dim]) for dim in range(len(matrix))]


def get_k_grid(k_axes):
    return np.array(np.meshgrid(k_axes[0], k_axes[1])).T


def get_kspace(k_axes, shapes, path, matrix, support, center, progress_bar=(tqdm, tqdm)):
    kspace = {}
    for tissue in progress_bar[0](shapes.keys(), desc='Tissues'):
        file = Path(path / tissue.replace(' ', '_')).with_suffix('.npy')
        if file.is_file():
            ksp = np.load(file)
            if ksp.shape == matrix:
                kspace[tissue] = ksp
        if tissue not in kspace:
            kspace[tissue] = np.zeros((matrix), dtype=complex)
            for shape in progress_bar[1](shapes[tissue], desc=f'"{tissue}" shapes', leave=False):
                kspace[tissue] += kspace_for_shape(shape, k_axes, get_k_grid(k_axes), support, center)
            np.save(file, kspace[tissue])
    return kspace


def load(name, min_voxel_size, progress_bar):
    path = Path(__file__).parent.resolve() / 'phantoms' / name
    phantom_object = {'name': name, 'path': path}
    for suffix in ['.toml', '.svg']:
        file = Path(path / name).with_suffix(suffix)
        if file.is_file():
            phantom_object['file'] = file
    if 'file' not in phantom_object:
        raise ValueError(f'Phantom {name}.svg/toml not found at {path}')
    phantom_object['shapes'] = load_shapes(phantom_object['file'])
    all_shapes = [shape for shapelist in phantom_object['shapes'].values() for shape in shapelist]
    phantom_object['support'], phantom_object['center'] = get_support_and_center(all_shapes)
    phantom_object['matrix'] = tuple(np.ceil(phantom_object['support'] / min_voxel_size).astype(int))
    phantom_object['k_axes'] = get_k_axes(phantom_object['matrix'], phantom_object['support'])
    phantom_object['kspace'] = get_kspace(phantom_object['k_axes'], phantom_object['shapes'], path, phantom_object['matrix'], phantom_object['support'], phantom_object['center'], progress_bar)
    return phantom_object


def get_phantom_names():
    phantoms = []
    phantom_path = Path(__file__).parent.resolve() / 'phantoms'
    for dir in phantom_path.iterdir():
        if dir.is_dir() and any(Path(dir / dir.name).with_suffix(suffix).is_file() for suffix in ['.toml', '.svg']):
            phantoms.append(dir.name)
    return phantoms