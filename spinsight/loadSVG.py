from spinsight import constants
import numpy as np
import xml.etree.ElementTree as ET
import re


def polygonArea(coords):
    return np.sum((coords[0]-np.roll(coords[0], 1)) * (coords[1]+np.roll(coords[1], 1))) / 2


def preparePath(path):
    if path[0] == path[-1]: 
        path.pop()
    return np.array(path).T


# Get coords from SVG path defined at https://www.w3.org/TR/SVG/paths.html
def getSubpaths(pathString, scale):
    supportedCommands = 'MZLHV'
    commands = supportedCommands + 'CSQTA'

    subpath, subpaths = [], []
    command = ''
    coord = (0, 0)
    x, y  = None, None
    for entry in pathString.strip().replace(',', ' ').split():
        if command.upper() == 'Z': # new subpath
            subpaths.append(preparePath(subpath))
            subpath = []
        if entry.upper() in commands:
            if entry.upper() in supportedCommands:
                command = entry
            else:
                raise Exception('Path command not supported: ' + entry)
        else: # no command; x or y coordinate
            if command.upper() == 'H':
                x, y = entry, 0
            elif command.upper() == 'V':
                x, y = 0, entry
            elif command.upper() in 'ML':
                if x is None:
                    x = entry
                else:
                    y = entry
            if x is not None and y is not None:
                relativeX = command.islower() or command.upper() == 'V'
                relativeY = command.islower() or command.upper() == 'H'
                coord = (float(y) * scale + coord[0] * relativeY, float(x) * scale + coord[1] * relativeX)
                subpath.append(coord)
                x, y  = None, None
    if command.upper() != 'Z':
        raise Exception('Warning: all paths must be closed')
    subpaths.append(preparePath(subpath))
    
    if sum([polygonArea(subpath) for subpath in subpaths]) < 0:
        for n in range(len(subpaths)):
            subpaths[n] = np.flip(subpaths[n], axis=1) # invert polygons to make total area positive
    return subpaths


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
    polygons = []
    for path in ET.parse(file).iter('{http://www.w3.org/2000/svg}path'): 
        style = parseStyleString(path.attrib['style'])
        hexcolor = style['fill'].strip('#')
        if hexcolor not in [v['hexcolor'] for v in constants.TISSUES.values()]:
            print('Warning: No tissue corresponding to hexcolor "{}" for path with id "{}"'.format(hexcolor, path.attrib['id']))
            continue
        tissue = [tissue for tissue in constants.TISSUES if constants.TISSUES[tissue]['hexcolor']==hexcolor][0]
        translation, rotation, scale = parseTransform(path.attrib['transform'] if 'transform' in path.attrib else '')
        if rotation != 0 or translation != (0, 0):
            raise NotImplementedError()
        subpaths = getSubpaths(path.attrib['d'], scale)
        for subpath in subpaths:
            polygons.append({'vertices': subpath, 'tissue': tissue})
    return polygons