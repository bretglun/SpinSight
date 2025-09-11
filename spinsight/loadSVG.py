from spinsight import constants
import numpy as np
import svgpathtools
import re
import warnings


def polygonArea(coords):
    return np.sum((coords[0]-np.roll(coords[0], 1)) * (coords[1]+np.roll(coords[1], 1))) / 2


def parseTransform(transformString):
    match = re.search(r'translate\((-?\d+.\d+)(px|%), (-?\d+.\d+)(px|%)\)', transformString)
    translation = (float(match.group(1)), float(match.group(3))) if match else (0, 0)

    match = re.search(r'rotate\((-?\d+.\d+)deg\)', transformString)
    rotation = float(match.group(1)) if match else 0
    
    match = re.search(r'scale\((-?\d+.\d+)\)', transformString)
    scale = float(match.group(1)) if match else 1

    return translation, rotation, scale


def parseStyleString(styleString):
    return {
        key.strip(): value.strip()
        for keyValue in styleString.split(';') if keyValue
        for key, value in [keyValue.split(':', 1)]
    }


def get_vertices(continuous_path):
    if not continuous_path.isclosed():
        raise Exception('All paths in SVG file must be closed')
    vertices = []
    for segment in continuous_path:
        if isinstance(segment, svgpathtools.path.Line):
            vertices.append((segment[0].imag, segment[0].real))
        else:
            warnings.warn('Only SVG line segments are supported, not {}'.format(type(segment)))
            return None
    if vertices:
        return np.array(vertices).T
    return None


# reads SVG file and returns polygon lists
def load(file):
    polygons = {}
    paths, attributes = svgpathtools.svg2paths(file)
    for p, path in enumerate(paths):
        attrib = attributes[p]
        style = parseStyleString(attrib['style'])
        hexcolor = style['fill'].strip('#')
        if hexcolor not in [v['hexcolor'] for v in constants.TISSUES.values()]:
            warnings.warn('No tissue corresponding to hexcolor "{}" for path with id "{}"'.format(hexcolor, attrib['id']))
            continue
        tissue = [tissue for tissue in constants.TISSUES if constants.TISSUES[tissue]['hexcolor']==hexcolor][0]
        if tissue not in polygons:
            polygons[tissue] = []
        translation, rotation, scale = parseTransform(attrib['transform'] if 'transform' in attrib else '')
        if rotation != 0 or translation != (0, 0):
            raise NotImplementedError()
        subpaths = path.continuous_subpaths()
        polys = []
        for subpath in subpaths:
            vertices = get_vertices(subpath)
            if vertices is not None:
                polys.append(vertices * scale)
        if sum([polygonArea(polygon) for polygon in polys]) < 0:
            # invert polygons to make total area positive
            polys = [np.flip(poly, axis=1) for poly in polys]
        polygons[tissue] += polys
    return polygons