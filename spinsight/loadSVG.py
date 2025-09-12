from spinsight import constants
import numpy as np
import xml.etree.ElementTree as ET
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
        return style['fill'].strip('#')
    return None


def get_styles(file):
    styles = {}
    style_element = ET.parse(file).getroot().find('{http://www.w3.org/2000/svg}style')
    if style_element is not None:
        for class_name, style_string in re.findall(r'\.(\w+)\s*\{([^}]*)\}', style_element.text):
            styles[class_name] = parse_style_string(style_string)
    return styles


# reads SVG file and returns polygon lists
def load(file):
    styles = get_styles(file)
    paths, attributes = svgpathtools.svg2paths(file)
    polygons = {}
    for p, path in enumerate(paths):
        attrib = attributes[p]
        hexcolor = get_hexcolor(attrib, styles)
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