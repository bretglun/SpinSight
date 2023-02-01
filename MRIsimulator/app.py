import holoviews as hv
import panel as pn
import param
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import re
import xarray as xr

hv.extension('bokeh')


TISSUES = {
    'gray':       {'PD': 1.0, 'T1': {'1.5T': 1100, '3T': 1330}, 'T2': {'1.5T':   92, '3T':   80}, 'hexcolor': '00ff00'},
    'white':      {'PD': 0.9, 'T1': {'1.5T':  560, '3T':  830}, 'T2': {'1.5T':   82, '3T':  110}, 'hexcolor': 'd40000'},
    'CSF':        {'PD': 1.0, 'T1': {'1.5T': 4280, '3T': 4160}, 'T2': {'1.5T': 2030, '3T': 2100}, 'hexcolor': '00ffff'},
    'fat':        {'PD': 1.0, 'T1': {'1.5T':  290, '3T':  370}, 'T2': {'1.5T':  165, '3T':  130}, 'hexcolor': 'ffe680'},
    'liver':      {'PD': 1.0, 'T1': {'1.5T':  586, '3T':  809}, 'T2': {'1.5T':   46, '3T':   34}, 'hexcolor': '800000'},
    'spleen':     {'PD': 1.0, 'T1': {'1.5T': 1057, '3T': 1328}, 'T2': {'1.5T':   79, '3T':   61}, 'hexcolor': 'ff0000'},
    'muscle':     {'PD': 1.0, 'T1': {'1.5T':  856, '3T':  898}, 'T2': {'1.5T':   27, '3T':   29}, 'hexcolor': '008000'},
    'kidneyMed':  {'PD': 1.0, 'T1': {'1.5T': 1412, '3T': 1545}, 'T2': {'1.5T':   85, '3T':   81}, 'hexcolor': 'aa4400'},
    'kidneyCor':  {'PD': 1.0, 'T1': {'1.5T':  966, '3T': 1142}, 'T2': {'1.5T':   87, '3T':   76}, 'hexcolor': '552200'},
    'spinalCord': {'PD': 1.0, 'T1': {'1.5T':  745, '3T':  993}, 'T2': {'1.5T':   74, '3T':   78}, 'hexcolor': 'ffff00'},
    'cortical':   {'PD': .05, 'T1': {'1.5T': 1000, '3T': 1000}, 'T2': {'1.5T':    3, '3T':    1}, 'hexcolor': '808000'},
    'blood':      {'PD': 1.0, 'T1': {'1.5T': 1441, '3T': 1932}, 'T2': {'1.5T':  290, '3T':  275}, 'hexcolor': 'ffffff'},
    'stomach':    {'PD': 0.2, 'T1': {'1.5T':  500, '3T':  500}, 'T2': {'1.5T':   30, '3T':   20}, 'hexcolor': '1a1a1a'},
    'perotineum': {'PD': 1.0, 'T1': {'1.5T':  900, '3T':  900}, 'T2': {'1.5T':   30, '3T':   30}, 'hexcolor': 'ff8080'},
}

SEQUENCES = ['Spin Echo', 'Spoiled Gradient Echo', 'Inversion Recovery']

PHANTOMS = {
    'abdomen': {'FOV': (320, 400), 'matrix': (512, 640)}
}

DIRECTIONS = {'anterior-posterior': 0, 'left-right': 1}

def polygonIsClockwise(coords):
    sum = 0
    for i in range(len(coords)):
        sum += (coords[i][0]-coords[i-1][0]) * (coords[i][1]+coords[i-1][1])
    return sum > 0

# Get coords from SVG path defined at https://www.w3.org/TR/SVG/paths.html
def getCoords(pathString, scale):
    supportedCommands = 'MZLHV'
    commands = supportedCommands + 'CSQTA'

    coords = []
    command = ''
    coord = (0, 0)
    x, y  = None, None
    for entry in pathString.strip().replace(',', ' ').split():
        if command.upper() == 'Z':
            raise Exception('Warning: unexpected entry following "{}"'.format(command))
        if entry.upper() in commands:
            if entry.upper() in supportedCommands:
                command = entry
            else:
                raise Exception('Path command not supported: ' + command)
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
                coord = (float(x) * scale + coord[0] * relativeX, float(y) * scale + coord[1] * relativeY)
                coords.append(coord)
                x, y  = None, None
    if coords[0] == coords[-1]: 
        coords.pop()
    if not polygonIsClockwise(coords):
        return coords[::-1]
    return coords


def parseTransform(transformString):
    match = re.search("translate\((-?\d+.\d+)(px|%), (-?\d+.\d+)(px|%)\)", transformString)
    translation = (float(match.group(1)), float(match.group(3))) if match else (0, 0)

    match = re.search("rotate\((-?\d+.\d+)deg\)", transformString)
    rotation = float(match.group(1)) if match else 0
    
    match = re.search("scale\((-?\d+.\d+)\)", transformString)
    scale = float(match.group(1)) if match else 1

    return translation, rotation, scale


# reads SVG file and returns polygon lists
def readSVG(inFile):
    polygons = []
    for path in ET.parse(inFile).iter('{http://www.w3.org/2000/svg}path'): 
        hexcolor = path.attrib['style'][6:12]
        tissue = [tissue for tissue in TISSUES if TISSUES[tissue]['hexcolor']==hexcolor][0]
        translation, rotation, scale = parseTransform(path.attrib['transform'] if 'transform' in path.attrib else '')
        if rotation != 0 or translation != (0, 0):
            raise NotImplementedError()
        polygons.append({('x', 'y'): getCoords(path.attrib['d'], scale), 'tissue': tissue})
    return polygons


def kspacePolygon(poly, phantom):
    # analytical 2D Fourier transform of polygon (see https://cvil.ucsd.edu/wp-content/uploads/2016/09/Realistic-analytical-polyhedral-MRI-phantoms.pdf)
    r = np.array(poly[('x', 'y')]) # position vectors of vertices Ve
    E = len(r) # number of vertices/edges
    Lv = np.roll(r, -1, axis=0) - r # edge vectors
    L = [np.linalg.norm(e) for e in Lv] # edge lengths
    t = [Lv[e]/L[e] for e in range(E)] # edge unit vectors
    n = np.roll(t, 1, axis=1)
    n[:,0] *= -1 # normals to tangents (pointing out from polygon)
    rc = r + Lv / 2 # position vector for center of edge

    ksp = np.zeros(np.product(phantom['matrix']), dtype=complex)
    coords = np.array([(u, v) for u in phantom['kAxes'][1] for v in phantom['kAxes'][0]])
    for e in range(E):
        arg = np.pi * np.dot(coords, Lv[e])
        zeroarg = arg==0
        nonzero = np.logical_not(zeroarg)
        ksp[nonzero] += L[e] * np.dot(coords[nonzero], n[e]) * np.sin(arg[nonzero]) / arg[nonzero] * np.exp(-2j*np.pi * np.dot(coords[nonzero], rc[e]))
        ksp[zeroarg] += L[e] * np.dot(coords[zeroarg], n[e]) * np.exp(-2j*np.pi * np.dot(coords[zeroarg], rc[e])) # sinc(0)=1
    ksp[1:] *= 1j / (2 * np.pi * np.linalg.norm(coords[1:], axis=1)**2)
    ksp[0] = abs(sum([r[e-1,0]*r[e,1] - r[e,0]*r[e-1,1] for e in range(E)]))/2 # kspace center equals polygon area
    return ksp.reshape(phantom['matrix'][::-1]).T # TODO: get x/y order right from start!


def resampleKspace(phantom, kAxes):
    kspace = {tissue: phantom['kspace'][tissue] for tissue in phantom['kspace']}
    for dim in range(len(kAxes)):
        sinc = np.sinc((np.tile(kAxes[dim], (len(phantom['kAxes'][dim]), 1)) - np.tile(phantom['kAxes'][dim][:, np.newaxis], (1, len(kAxes[dim])))) * phantom['FOV'][dim])
        for tissue in phantom['kspace']:
            kspace[tissue] = np.moveaxis(np.tensordot(kspace[tissue], sinc, axes=(dim, 0)), -1, dim)
    return kspace


def zerofill(kspace, reconMatrix):
    shape = [1] * kspace.ndim
    for dim, n in enumerate(kspace.shape):
        shape[0] = reconMatrix[dim] - n
        kspace = np.insert(kspace, n-n//2, np.zeros(tuple(shape)), axis=dim)
    return kspace


def getPixelShiftMatrix(matrix, shift):
        phase = [np.fft.fftfreq(matrix[dim]) * shift[dim] * 2*np.pi for dim in range(len(matrix))]
        return np.exp(1j * np.sum(np.stack(np.meshgrid(*phase[::-1])), axis=0))


def crop(arr, shape):
    # Crop array from center according to shape
    for dim, n in enumerate(arr.shape):
        arr = arr.take(np.array(range(shape[dim])) + (n-shape[dim])//2, dim)
    return arr


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
    object = param.ObjectSelector(default=list(PHANTOMS.keys())[0], objects=PHANTOMS.keys())
    fieldStrength = param.ObjectSelector(default='1.5T', objects=['1.5T', '3T'])
    sequence = param.ObjectSelector(default=SEQUENCES[0], objects=SEQUENCES)
    TR = param.Number(default=1000.0, bounds=(0, 5000.0))
    TE = param.Number(default=1.0, bounds=(0, 100.0))
    FA = param.Number(default=90.0, bounds=(0, 90.0), precedence=-1)
    TI = param.Number(default=1.0, bounds=(0, 1000.0), precedence=-1)
    FOVX = param.Number(default=420, bounds=(100, 600))
    FOVY = param.Number(default=420, bounds=(100, 600))
    matrixX = param.Integer(default=128, bounds=(16, 600))
    matrixY = param.Integer(default=128, bounds=(16, 600))
    reconMatrixX = param.Integer(default=256, bounds=(matrixX.default, 1024))
    reconMatrixY = param.Integer(default=256, bounds=(matrixY.default, 1024))
    freqeuencyDirection = param.ObjectSelector(default=list(DIRECTIONS.keys())[-1], objects=DIRECTIONS.keys())
    

    def __init__(self, **params):
        super().__init__(**params)
        self.loadPhantom()
        self.sampleKspace()
        self.reconstruct()
        self.updateMagnetization()
    

    @param.depends('sequence', watch=True)
    def _updateVisibility(self):
        self.param.FA.precedence = 1 if self.sequence=='Spoiled Gradient Echo' else -1
        self.param.TI.precedence = 1 if self.sequence=='Inversion Recovery' else -1
    

    @param.depends('matrixX', watch=True)
    def _updateReconMatrixXbounds(self):
        self.param.reconMatrixX.bounds = (self.matrixX, self.param.reconMatrixX.bounds[1])
        self.reconMatrixX = max(self.reconMatrixX, self.matrixX)    


    @param.depends('matrixY', watch=True)
    def _updateReconMatrixYbounds(self):
        self.param.reconMatrixY.bounds = (self.matrixY, self.param.reconMatrixY.bounds[1])
        self.reconMatrixY = max(self.reconMatrixY, self.matrixY)

    
    @param.depends('object', watch=True)
    def loadPhantom(self):
        phantomPath = Path('./phantoms/{p}'.format(p=self.object))
        self.phantom = PHANTOMS[self.object]
        self.phantom['kAxes'] = [np.fft.fftfreq(self.phantom['matrix'][dim]) * self.phantom['matrix'][dim] / self.phantom['FOV'][dim] for dim in range(len(self.phantom['matrix']))]
        npyFiles = list(phantomPath.glob('*.npy'))
        if npyFiles:
            self.tissues = set()
            self.phantom['kspace'] = {}
            for file in npyFiles:
                tissue = file.stem
                self.tissues.add(tissue)
                self.phantom['kspace'][tissue] = np.load(file)
        else:
            polys = readSVG(Path(phantomPath / self.object).with_suffix('.svg'))
            self.tissues = set([poly['tissue'] for poly in polys])
            self.phantom['kspace'] = {tissue: np.zeros((self.phantom['matrix']), dtype=complex) for tissue in self.tissues}
            for poly in polys:
                self.phantom['kspace'][poly['tissue']] += kspacePolygon(poly, self.phantom)
            for tissue in self.tissues:
                file = Path(phantomPath / tissue).with_suffix('.npy')
                np.save(file, self.phantom['kspace'][tissue])
    

    @param.depends('object', 'matrixX', 'matrixY', 'reconMatrixX', 'reconMatrixY', 'FOVX', 'FOVY', 'freqeuencyDirection', watch=True)
    def sampleKspace(self):
        self.matrix = [self.matrixY, self.matrixX]
        self.reconMatrix = [self.reconMatrixY, self.reconMatrixX]
        self.FOV = [self.FOVY, self.FOVX]
        self.freqDir = DIRECTIONS[self.freqeuencyDirection]
        
        self.oversampledMatrix = self.matrix.copy() # account for oversampling in frequency encoding direction
        self.oversampledReconMatrix = self.reconMatrix.copy()
        if self.FOV[self.freqDir] < self.phantom['FOV'][self.freqDir]: # At least Nyquist sampling in frequency encoding direction
            self.oversampledMatrix[self.freqDir] = int(np.ceil(self.phantom['FOV'][self.freqDir] * self.matrix[self.freqDir] / self.FOV[self.freqDir]))
            self.oversampledReconMatrix[self.freqDir] = int(self.reconMatrix[self.freqDir] * self.oversampledMatrix[self.freqDir] / self.matrix[self.freqDir])
        
        self.kAxes = [np.fft.fftfreq(self.oversampledMatrix[dim]) * self.matrix[dim] / self.FOV[dim] for dim in range(len(self.matrix))]
        self.kspace = resampleKspace(self.phantom, self.kAxes)


    @param.depends('object', 'matrixX', 'matrixY', 'reconMatrixX', 'reconMatrixY', 'FOVX', 'FOVY', 'freqeuencyDirection', watch=True)
    def reconstruct(self):
        self.imageArrays = {}
        # half pixel shift for even dims (based on reconMatrix, not oversampledReconMatrix!)
        shift = [.0 if self.reconMatrix[dim]%2 else .5 for dim in range(len(self.reconMatrix))]
        halfPixelShift = getPixelShiftMatrix(self.oversampledReconMatrix, shift)
        for tissue in self.tissues:
            self.kspace[tissue] = zerofill(self.kspace[tissue], self.oversampledReconMatrix)
            self.kspace[tissue] *= halfPixelShift
            self.imageArrays[tissue] = np.fft.fftshift(np.fft.ifft2(self.kspace[tissue]))
            self.imageArrays[tissue] = crop(self.imageArrays[tissue], self.reconMatrix)
        self.iAxes = [(np.arange(self.reconMatrix[dim]) - (self.reconMatrix[dim]-1)/2) / self.reconMatrix[dim] * self.FOV[dim] for dim in range(2)]


    @param.depends('object', 'fieldStrength', 'sequence', 'TR', 'TE', 'FA', 'TI', watch=True)
    def updateMagnetization(self):
        self.M = {tissue: getM(tissue, self.sequence, self.TR, self.TE, self.TI, self.FA, self.fieldStrength) for tissue in self.tissues}


    @param.depends('object', 'fieldStrength', 'matrixX', 'matrixY', 'reconMatrixX', 'reconMatrixY', 'FOVX', 'FOVY', 'freqeuencyDirection', 'sequence', 'TR', 'TE', 'FA', 'TI')
    def getImage(self):
        pixelArray = np.zeros(self.reconMatrix, dtype=complex)
        for tissue in self.tissues:
            pixelArray += self.imageArrays[tissue] * self.M[tissue]
        
        img = xr.DataArray(
            np.abs(pixelArray), 
            dims=('y', 'x'),
            coords={'x': self.iAxes[1], 'y': self.iAxes[0][::-1]}
        )
        img.x.attrs['units'] = img.y.attrs['units'] = 'mm'

        return hv.Image(img, vdims=['magnitude']).options(cmap='gray', aspect='equal')


explorer = MRIsimulator(name='')
title = '# MRI simulator'
author = '*Written by [Johan Berglund](mailto:johan.berglund@akademiska.se), Ph.D.*'
contrastParams = pn.panel(explorer.param, parameters=['fieldStrength', 'sequence', 'TR', 'TE', 'FA', 'TI'], name='Contrast')
geometryParams = pn.panel(explorer.param, parameters=['FOVX', 'FOVY', 'matrixX', 'matrixY', 'reconMatrixX', 'reconMatrixY', 'freqeuencyDirection'], name='Geometry')
dmapMRimage = hv.DynamicMap(explorer.getImage).opts(frame_height=500)
dashboard = pn.Row(pn.Column(pn.pane.Markdown(title), pn.Row(contrastParams, geometryParams), pn.pane.Markdown(author)), dmapMRimage)
dashboard.servable() # run by ´panel serve app.py´, then open http://localhost:5006/app in browser

# TODO: add BW param
# TODO: add kpsace dephasing during readout
# TODO: do T2(*)-weighting in kpsace
# TODO: add noise
# TODO: add fatsat
# TODO: bounds on acq params
# TODO: handle fat shift (including signal cancellation)
# TODO: add ACQ time
# TODO: add k-space plot
# TODO: add params for matrix/pixelSize and BW like different vendors and handle their correlation
# TODO: add apodization
# TODO: parallel imaging (GRAPPA)
# TODO: B0 inhomogeneity
# TODO: T2* weighting
# TODO: Fast spin echo
# TODO: EPI