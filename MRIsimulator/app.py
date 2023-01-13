import holoviews as hv
import panel as pn
import param
import numpy as np
import xml.etree.ElementTree as ET
import re

hv.extension('bokeh')


TISSUES = {
    'gray':     {'PD': 1.0, 'T1': {'1.5T': 1100, '3T': 1330}, 'T2': {'1.5T':   92, '3T':   80}, 'hexcolor': '00ff00'},
    'white':    {'PD': 0.9, 'T1': {'1.5T':  560, '3T':  830}, 'T2': {'1.5T':   82, '3T':  110}, 'hexcolor': 'd40000'},
    'CSF':      {'PD': 1.0, 'T1': {'1.5T': 4280, '3T': 4160}, 'T2': {'1.5T': 2030, '3T': 2100}, 'hexcolor': '000001'},
    'fat':      {'PD': 1.0, 'T1': {'1.5T':  290, '3T':  370}, 'T2': {'1.5T':  165, '3T':  130}, 'hexcolor': '000002'}
}


SEQUENCES = ['Spin Echo', 'Spoiled Gradient Echo', 'Inversion Recovery']


def readSVG(inFile):
    # reads SVG file and returns polygon lists
    polygons = []
    for path in ET.parse(inFile).iter('{http://www.w3.org/2000/svg}path'): 
        hexcolor = path.attrib['style'][6:12]
        tissue = [tissue for tissue in TISSUES if TISSUES[tissue]['hexcolor']==hexcolor][0]
        coords = re.findall(r'(-?\d+\.?\d*)\,(-?\d+\.?\d*)', path.attrib['d'])
        coords = [(float(x), float(y)) for (x,y) in coords]
        polygons.append({('x', 'y'): coords, 'tissue': tissue, 'M': .5})
    return polygons

    # TODO: better understand svg path format


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
        img = hv.Polygons(self.polys, vdims='M').options(aspect='equal', cmap='gray').redim.range(M=(0,1)).opts(invert_yaxis=True)
        return img


explorer = MRIsimulator('abdomen.svg', name='MR Contrast Explorer')
dmapMRimage = hv.DynamicMap(explorer.getImage).opts(framewise=True, frame_height=300)
dashboard = pn.Column(pn.Row(pn.panel(explorer.param), dmapMRimage))
dashboard.servable() # run by ´panel serve app.py´, then open http://localhost:XXXX/app in browser