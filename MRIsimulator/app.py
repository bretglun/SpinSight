import holoviews as hv
import panel as pn
import param
import numpy as np
import xml.etree.ElementTree as ET

hv.extension('bokeh')


TISSUES = {
    'gray':       {'PD': 1.0, 'T1': {'1.5T': 1100, '3T': 1330}, 'T2': {'1.5T':   92, '3T':   80}, 'hexcolor': '00ff00'},
    'white':      {'PD': 0.9, 'T1': {'1.5T':  560, '3T':  830}, 'T2': {'1.5T':   82, '3T':  110}, 'hexcolor': 'd40000'},
    'CSF':        {'PD': 1.0, 'T1': {'1.5T': 4280, '3T': 4160}, 'T2': {'1.5T': 2030, '3T': 2100}, 'hexcolor': '00ffff'},
    'fat':        {'PD': 1.0, 'T1': {'1.5T':  290, '3T':  370}, 'T2': {'1.5T':  165, '3T':  130}, 'hexcolor': 'ffe680'},
    'liver':      {'PD': 1.0, 'T1': {'1.5T':  586, '3T':  809}, 'T2': {'1.5T':   46, '3T':   34}, 'hexcolor': '800000'},
    'spleen':     {'PD': 1.0, 'T1': {'1.5T': 1057, '3T': 1328}, 'T2': {'1.5T':   79, '3T':   61}, 'hexcolor': 'ff0000'},
    'muscle':     {'PD': 1.0, 'T1': {'1.5T':  856, '3T':  898}, 'T2': {'1.5T':   27, '3T':   29}, 'hexcolor': '008000'},
    'kidney med': {'PD': 1.0, 'T1': {'1.5T': 1412, '3T': 1545}, 'T2': {'1.5T':   85, '3T':   81}, 'hexcolor': 'aa4400'},
    'kidney cor': {'PD': 1.0, 'T1': {'1.5T':  966, '3T': 1142}, 'T2': {'1.5T':   87, '3T':   76}, 'hexcolor': '552200'},
    'spinal cord':{'PD': 1.0, 'T1': {'1.5T':  745, '3T':  993}, 'T2': {'1.5T':   74, '3T':   78}, 'hexcolor': 'ffff00'},
    'cortical':   {'PD': .05, 'T1': {'1.5T': 1000, '3T': 1000}, 'T2': {'1.5T':    3, '3T':    1}, 'hexcolor': '808000'},
    'blood':      {'PD': 1.0, 'T1': {'1.5T': 1441, '3T': 1932}, 'T2': {'1.5T':  290, '3T':  275}, 'hexcolor': 'ffffff'},
    'stomach':    {'PD': 0.2, 'T1': {'1.5T':  500, '3T':  500}, 'T2': {'1.5T':   30, '3T':   20}, 'hexcolor': '1a1a1a'},
    'perotineum': {'PD': 1.0, 'T1': {'1.5T':  900, '3T':  900}, 'T2': {'1.5T':   30, '3T':   30}, 'hexcolor': 'ff8080'},
}


SEQUENCES = ['Spin Echo', 'Spoiled Gradient Echo', 'Inversion Recovery']


# Get coords from SVG path defined at https://www.w3.org/TR/SVG/paths.html
def getCoords(pathString):
    supportedCommands = 'MZLHV'
    commands = supportedCommands + 'CSQTA'

    coords = []
    command = None
    coord = (0, 0)
    for entry in pathString.strip().split():
        if entry.upper() in commands:
            if entry.upper() in supportedCommands:
                command = entry
            else:
                raise Exception('Path command not supported: ' + command)
        else:
            if command.upper() == 'H':
                x, y = entry, 0
            elif command.upper() == 'V':
                x, y = 0, entry
            elif command.upper() in 'ML':
                x, y = entry.split(',')
            elif command.upper() == 'Z':
                raise Exception('Warning: unexpected "{}" followed by number'.format(command))
            coord = (float(x) + coord[0] * command.islower(), float(y) + coord[1] * command.islower())
            coords.append(coord)
    return coords


# reads SVG file and returns polygon lists
def readSVG(inFile):
    polygons = []
    for path in ET.parse(inFile).iter('{http://www.w3.org/2000/svg}path'): 
        hexcolor = path.attrib['style'][6:12]
        tissue = [tissue for tissue in TISSUES if TISSUES[tissue]['hexcolor']==hexcolor][0]
        polygons.append({('x', 'y'): getCoords(path.attrib['d']), 'tissue': tissue})
    return polygons


def getM(tissue, seqType, TR, TE, TI, FA, B0='1.5T'):
    PD = TISSUES[tissue]['PD']
    T1 = TISSUES[tissue]['T1'][B0]
    T2 = TISSUES[tissue]['T2'][B0]
    
    E1 = np.exp(-TR/T1)
    E2 = np.exp(-TE/T2)
    if seqType == 'Spin Echo':
        return PD * (1 - E1) * E2
    elif seqType == 'Spoiled Gradient Echo':
        return PD * E2 * np.sin(np.radians(FA)) * (1 - E1) / (1 - np.cos(np.radians(FA)) * E1)
    elif seqType == 'Inversion Recovery':
        return PD * E2 * (1 - 2 * np.exp(-TI/T1) + E1)
    else:
        raise Exception('Unknown sequence type: {}'.format(seqType))


class MRIsimulator(param.Parameterized):
    sequence = param.ObjectSelector(default=SEQUENCES[0], objects=SEQUENCES)
    TR = param.Number(default=1000.0, bounds=(0, 5000.0))
    TE = param.Number(default=1.0, bounds=(0, 100.0))
    FA = param.Number(default=90.0, bounds=(0, 90.0), precedence=-1)
    TI = param.Number(default=1.0, bounds=(0, 1000.0), precedence=-1)
    

    def __init__(self, SVGfile, **params):
        super().__init__(**params)
        self.polys = readSVG(SVGfile)
        self.tissues = set([poly['tissue'] for poly in self.polys])
    

    @param.depends('sequence', watch=True)
    def _updateVisibility(self):
        self.param.FA.precedence = 1 if self.sequence=='Spoiled Gradient Echo' else -1
        self.param.TI.precedence = 1 if self.sequence=='Inversion Recovery' else -1


    @param.depends('sequence', 'TR', 'TE', 'FA', 'TI')
    def getImage(self):
        # calculate and update magnetization
        M = {tissue: getM(tissue, self.sequence, self.TR, self.TE, self.TI, self.FA) for tissue in self.tissues}
        for i, _ in enumerate(self.polys):
            self.polys[i]['M'] = M[self.polys[i]['tissue']]
        img = hv.Polygons(self.polys, vdims='M').options(aspect='equal', cmap='gray').redim.range(M=(0,1)).opts(invert_xaxis=False, invert_yaxis=True, bgcolor='black')
        return img


explorer = MRIsimulator('abdomen.svg', name='MR Contrast Explorer')
dmapMRimage = hv.DynamicMap(explorer.getImage).opts(framewise=True, frame_height=300)
dashboard = pn.Column(pn.Row(pn.panel(explorer.param), dmapMRimage))
dashboard.servable() # run by ´panel serve app.py´, then open http://localhost:5006/app in browser