from spinsight import constants
import numpy as np
import svgpathtools
import re


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


# reads SVG file and returns polygon lists
def load(file):
    polygons = {}
    paths, attributes = svgpathtools.svg2paths(file)
    for p, path in enumerate(paths):
        attrib = attributes[p]
        style = parseStyleString(attrib['style'])
        hexcolor = style['fill'].strip('#')
        if hexcolor not in [v['hexcolor'] for v in constants.TISSUES.values()]:
            print('Warning: No tissue corresponding to hexcolor "{}" for path with id "{}"'.format(hexcolor, path.attrib['id']))
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
            if not subpath.isclosed():
                raise Exception('All paths in SVG file must be closed')
            polys.append(np.array([(p[0].imag * scale, p[0].real * scale) for p in subpath]).T)
        if sum([polygonArea(polygon) for polygon in polys]) < 0:
            # invert polygons to make total area positive
            polys = [np.flip(poly, axis=1) for poly in polys]
        polygons[tissue] += polys
    return polygons