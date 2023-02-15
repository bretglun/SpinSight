import holoviews as hv
import panel as pn
import param
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import re
import xarray as xr

hv.extension('bokeh')


GYRO = 42.577 # 1H gyromagnetic ratio [MHz/T]

TISSUES = {
    'gray':       {'PD': 1.0, 'T1': {1.5: 1100, 3.0: 1330}, 'T2': {1.5:   92, 3.0:   80}, 'hexcolor': '00ff00'},
    'white':      {'PD': 0.9, 'T1': {1.5:  560, 3.0:  830}, 'T2': {1.5:   82, 3.0:  110}, 'hexcolor': 'd40000'},
    'CSF':        {'PD': 1.0, 'T1': {1.5: 4280, 3.0: 4160}, 'T2': {1.5: 2030, 3.0: 2100}, 'hexcolor': '00ffff'},
    'adipose':    {'PD': 1.0, 'T1': {1.5:  290, 3.0:  370}, 'T2': {1.5:  165, 3.0:  130}, 'hexcolor': 'ffe680'},
    'liver':      {'PD': 1.0, 'T1': {1.5:  586, 3.0:  809}, 'T2': {1.5:   46, 3.0:   34}, 'hexcolor': '800000'},
    'spleen':     {'PD': 1.0, 'T1': {1.5: 1057, 3.0: 1328}, 'T2': {1.5:   79, 3.0:   61}, 'hexcolor': 'ff0000'},
    'muscle':     {'PD': 1.0, 'T1': {1.5:  856, 3.0:  898}, 'T2': {1.5:   27, 3.0:   29}, 'hexcolor': '008000'},
    'kidneyMed':  {'PD': 1.0, 'T1': {1.5: 1412, 3.0: 1545}, 'T2': {1.5:   85, 3.0:   81}, 'hexcolor': 'aa4400'},
    'kidneyCor':  {'PD': 1.0, 'T1': {1.5:  966, 3.0: 1142}, 'T2': {1.5:   87, 3.0:   76}, 'hexcolor': '552200'},
    'spinalCord': {'PD': 1.0, 'T1': {1.5:  745, 3.0:  993}, 'T2': {1.5:   74, 3.0:   78}, 'hexcolor': 'ffff00'},
    'cortical':   {'PD': .05, 'T1': {1.5: 1000, 3.0: 1000}, 'T2': {1.5:    3, 3.0:    1}, 'hexcolor': '808000'},
    'blood':      {'PD': 1.0, 'T1': {1.5: 1441, 3.0: 1932}, 'T2': {1.5:  290, 3.0:  275}, 'hexcolor': 'ffffff'},
    'stomach':    {'PD': 0.2, 'T1': {1.5:  500, 3.0:  500}, 'T2': {1.5:   30, 3.0:   20}, 'hexcolor': '1a1a1a'},
    'perotineum': {'PD': 1.0, 'T1': {1.5:  900, 3.0:  900}, 'T2': {1.5:   30, 3.0:   30}, 'hexcolor': 'ff8080'},
}

ADIPOSERESONANCES = { 'adiposeWater': {'shift': 4.7 - 4.7, 'ratio': .050, 'ratioWithFatSat': .050},
                      'adiposeFat1':  {'shift': 0.9 - 4.7, 'ratio': .083, 'ratioWithFatSat': .010},
                      'adiposeFat2':  {'shift': 1.3 - 4.7, 'ratio': .659, 'ratioWithFatSat': .033},
                      'adiposeFat3':  {'shift': 2.1 - 4.7, 'ratio': .122, 'ratioWithFatSat': .038},
                      'adiposeFat4':  {'shift': 2.8 - 4.7, 'ratio': .004, 'ratioWithFatSat': .003},
                      'adiposeFat5':  {'shift': 4.3 - 4.7, 'ratio': .037, 'ratioWithFatSat': .037}, 
                      'adiposeFat6':  {'shift': 5.3 - 4.7, 'ratio': .045, 'ratioWithFatSat': .045}}

SEQUENCES = ['Spin Echo', 'Spoiled Gradient Echo', 'Inversion Recovery']

DURATIONS = {'exc': 1.0, 'ref': 4.0, 'inv': 4.0, 'spoil': 1.0} # excitation, refocusing, inversion, spoiling [msec]

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


def getT2w(tissue, decayTime, dephasingTime, B0):
    T2 = TISSUES[tissue]['T2'][B0]
    T2prim = 100. # ad hoc value [msec]
    E2 = np.exp(-decayTime/T2)
    E2prim = np.exp(-np.abs(dephasingTime)/T2prim)
    return E2 * E2prim


def getPDandT1w(tissue, seqType, TR, TE, TI, FA, B0):
    PD = TISSUES[tissue]['PD']
    T1 = TISSUES[tissue]['T1'][B0]
    
    E1 = np.exp(-TR/T1)
    if seqType == 'Spin Echo':
        return PD * (1 - E1)
    elif seqType == 'Spoiled Gradient Echo':
        return PD * np.sin(np.radians(FA)) * (1 - E1) / (1 - np.cos(np.radians(FA)) * E1)
    elif seqType == 'Inversion Recovery':
        return PD * (1 - 2 * np.exp(-TI/T1) + E1)
    else:
        raise Exception('Unknown sequence type: {}'.format(seqType))


class MRIsimulator(param.Parameterized):
    object = param.ObjectSelector(default=list(PHANTOMS.keys())[0], objects=PHANTOMS.keys(), label='Phantom object')
    fieldStrength = param.ObjectSelector(default=1.5, objects=[1.5, 3.0], label='B0 field strength [T]')
    sequence = param.ObjectSelector(default=SEQUENCES[0], objects=SEQUENCES, label='Pulse sequence')
    FatSat = param.Boolean(default=False, label='Fat saturation')
    TR = param.Number(default=750.0, bounds=(0, 8000.0), label='TR [msec]')
    shortTRrange = param.Boolean(default=False, label='Short TR range')
    TE = param.Number(default=10.0, bounds=(0, 400.0), label='TE [msec]')
    shortTErange = param.Boolean(default=False, label='Short TE range')
    FA = param.Number(default=90.0, bounds=(1, 90.0), precedence=-1, label='Flip angle [°]')
    TI = param.Number(default=50.0, bounds=(0, 4000.0), precedence=-1, label='TI [msec]')
    FOVX = param.Number(default=420, bounds=(100, 600), label='FOV x [mm]')
    FOVY = param.Number(default=420, bounds=(100, 600), label='FOV y [mm]')
    matrixX = param.Integer(default=128, bounds=(16, 600), label='Acquisition matrix x')
    matrixY = param.Integer(default=128, bounds=(16, 600), label='Acquisition matrix y')
    reconMatrixX = param.Integer(default=256, bounds=(matrixX.default, 1024), label='Reconstruction matrix x')
    reconMatrixY = param.Integer(default=256, bounds=(matrixY.default, 1024), label='Reconstruction matrix y')
    freqeuencyDirection = param.ObjectSelector(default=list(DIRECTIONS.keys())[-1], objects=DIRECTIONS.keys(), label='Frequency encoding direction')
    pixelBandWidth = param.Number(default=500, bounds=(50, 1000), label='Pixel bandwidth [Hz]')
    NSA = param.Integer(default=1, bounds=(1, 32), label='NSA')
    

    def __init__(self, **params):
        super().__init__(**params)
        self.updateReadoutDuration()
        self.updateTEbounds()
        self.updateTRbounds()
        self.loadPhantom()
        self.sampleKspace()
        self.updateSamplingTime()
        self.updateKspaceModulation()
        self.modulateKspace()
        self.addNoise()
        self.reconstruct()
        self.updatePDandT1w()
    

    @param.depends('pixelBandWidth', watch=True)
    def updateReadoutDuration(self):
        self.readoutDuration = 1e3 / self.pixelBandWidth # [msec]
    
    
    @param.depends('shortTRrange', 'pixelBandWidth', 'sequence', 'TE', 'TI', watch=True)
    def updateTRbounds(self):
        minTR = DURATIONS['exc']/2 + self.TE + self.readoutDuration/2 + DURATIONS['spoil']
        if self.sequence == 'Inversion Recovery': minTR += DURATIONS['inv']/2 + self.TI - DURATIONS['exc']/2
        maxTR = max(minTR, 100 if self.shortTRrange else 8000)
        self.param.TR.bounds = (minTR, maxTR)
        self.TR = min(max(self.TR, minTR), maxTR)
    
    
    @param.depends('shortTErange', 'pixelBandWidth', 'sequence', 'TR', 'TI', watch=True)
    def updateTEbounds(self):
        if self.sequence in ['Spin Echo', 'Inversion Recovery']: # seqs with refocusing pulse
            minTE = max((DURATIONS['exc'] + DURATIONS['ref']), (DURATIONS['ref'] + self.readoutDuration))
        else:
            minTE = (DURATIONS['exc'] + self.readoutDuration)/2
        maxTE = self.TR - (DURATIONS['exc']/2 + self.readoutDuration/2 + DURATIONS['spoil'])
        if self.sequence == 'Inversion Recovery':
            maxTE += DURATIONS['exc']/2 - DURATIONS['inv']/2 - self.TI
        maxTE = min(maxTE, 25 if self.shortTErange else 400)
        self.param.TE.bounds = (minTE, maxTE)
        self.TE = min(max(self.TE, minTE), maxTE)
    

    @param.depends('pixelBandWidth', 'sequence', 'TE', 'TR', watch=True)
    def updateTIbounds(self):
        if self.sequence != 'Inversion Recovery': return
        minTI = (DURATIONS['inv'] + DURATIONS['exc'])/2
        maxTI = self.TR - (DURATIONS['inv']/2 + self.TE + self.readoutDuration/2 + DURATIONS['spoil'])
        maxTI = min(4000, maxTI)
        self.param.TI.bounds = (minTI, maxTI)
        self.TI = min(max(self.TI, minTI), maxTI)

    
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
        phantomPath = Path(__file__).parent.resolve() / 'phantoms/{p}'.format(p=self.object)
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
        self.plainKspace = resampleKspace(self.phantom, self.kAxes)
    

    @param.depends('object', 'matrixX', 'matrixY', 'reconMatrixX', 'reconMatrixY', 'FOVX', 'FOVY', 'freqeuencyDirection', 'fieldStrength', 'pixelBandWidth', 'NSA', watch=True)
    def updateSamplingTime(self):
        self.samplingTime = np.fft.fftfreq(len(self.kAxes[self.freqDir])) / self.pixelBandWidth * 1e3 # msec
        self.noiseStd = 10. / np.sqrt(np.diff(self.samplingTime[:2]) * self.NSA) / self.fieldStrength
        self.samplingTime = np.expand_dims(self.samplingTime, axis=[dim for dim in range(len(self.matrix)) if dim != self.freqDir])
    

    @param.depends('object', 'matrixX', 'matrixY', 'reconMatrixX', 'reconMatrixY', 'FOVX', 'FOVY', 'freqeuencyDirection', 'TE', 'fieldStrength', 'pixelBandWidth', 'sequence', watch=True)
    def updateKspaceModulation(self):
        decayTime = self.samplingTime + self.TE
        dephasingTime = decayTime if 'Gradient Echo' in self.sequence else self.samplingTime
        
        self.kspaceModulation = {}
        for tissue in self.tissues:
            if tissue != 'adipose':
                self.kspaceModulation[tissue] = getT2w(tissue, decayTime, dephasingTime, self.fieldStrength)
            else:
                T2w = getT2w(tissue, decayTime, dephasingTime, self.fieldStrength)
                for component, resonance in ADIPOSERESONANCES.items():
                    dephasing = np.exp(2j*np.pi * GYRO * self.fieldStrength * resonance['shift'] * dephasingTime * 1e-3)
                    self.kspaceModulation[component] = dephasing * T2w


    @param.depends('object', 'matrixX', 'matrixY', 'reconMatrixX', 'reconMatrixY', 'FOVX', 'FOVY', 'freqeuencyDirection', 'TE', 'fieldStrength', 'pixelBandWidth', 'sequence', watch=True)
    def modulateKspace(self):
        self.kspace = {}
        for tissue in self.tissues:
            if tissue != 'adipose':
                self.kspace[tissue] = self.plainKspace[tissue] * self.kspaceModulation[tissue]
            else:
                for component in ADIPOSERESONANCES:
                    self.kspace[component] = self.plainKspace[tissue] * self.kspaceModulation[component]
    
    
    @param.depends('object', 'matrixX', 'matrixY', 'reconMatrixX', 'reconMatrixY', 'FOVX', 'FOVY', 'freqeuencyDirection', 'TE', 'fieldStrength', 'pixelBandWidth', 'NSA', 'sequence', watch=True)
    def addNoise(self):
        self.kspace['noise'] = np.random.normal(0, self.noiseStd, self.oversampledMatrix) + 1j * np.random.normal(0, self.noiseStd, self.oversampledMatrix)


    @param.depends('object', 'matrixX', 'matrixY', 'reconMatrixX', 'reconMatrixY', 'FOVX', 'FOVY', 'freqeuencyDirection', 'TE', 'fieldStrength', 'pixelBandWidth', 'NSA', 'sequence', watch=True)
    def reconstruct(self):
        self.imageArrays = {}
        # half pixel shift for even dims (based on reconMatrix, not oversampledReconMatrix!)
        shift = [.0 if self.reconMatrix[dim]%2 else .5 for dim in range(len(self.reconMatrix))]
        halfPixelShift = getPixelShiftMatrix(self.oversampledReconMatrix, shift)
        for component in self.kspace:
            kspace = zerofill(self.kspace[component], self.oversampledReconMatrix)
            kspace *= halfPixelShift
            self.imageArrays[component] = np.fft.fftshift(np.fft.ifft2(kspace))
            self.imageArrays[component] = crop(self.imageArrays[component], self.reconMatrix)
        self.iAxes = [(np.arange(self.reconMatrix[dim]) - (self.reconMatrix[dim]-1)/2) / self.reconMatrix[dim] * self.FOV[dim] for dim in range(2)]


    @param.depends('object', 'fieldStrength', 'sequence', 'TR', 'TE', 'FA', 'TI', watch=True)
    def updatePDandT1w(self):
        self.PDandT1w = {tissue: getPDandT1w(tissue, self.sequence, self.TR, self.TE, self.TI, self.FA, self.fieldStrength) for tissue in self.tissues}


    @param.depends('object', 'fieldStrength', 'matrixX', 'matrixY', 'reconMatrixX', 'reconMatrixY', 'FOVX', 'FOVY', 'freqeuencyDirection', 'pixelBandWidth', 'NSA', 'sequence', 'TR', 'TE', 'FA', 'TI', 'FatSat')
    def getImage(self):
        pixelArray = np.zeros(self.reconMatrix, dtype=complex)
        for component in self.imageArrays:
            if component=='noise':
                continue
            elif 'adipose' in component: 
                tissue = 'adipose'
                ratio = ADIPOSERESONANCES[component]['ratioWithFatSat' if self.FatSat else 'ratio']
            else:
                tissue = component
                ratio = 1.0
            pixelArray += self.imageArrays[component] * self.PDandT1w[tissue] * ratio
        pixelArray += self.imageArrays['noise']
        
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
contrastParams = pn.panel(explorer.param, parameters=['fieldStrength', 'sequence', 'FatSat', 'TR', 'shortTRrange', 'TE', 'shortTErange', 'FA', 'TI'], name='Contrast')
geometryParams = pn.panel(explorer.param, parameters=['FOVX', 'FOVY', 'matrixX', 'matrixY', 'reconMatrixX', 'reconMatrixY', 'freqeuencyDirection', 'pixelBandWidth', 'NSA'], name='Geometry')
dmapMRimage = hv.DynamicMap(explorer.getImage).opts(frame_height=500)
dashboard = pn.Row(pn.Column(pn.pane.Markdown(title), pn.Row(contrastParams, geometryParams), pn.pane.Markdown(author)), dmapMRimage)
dashboard.servable() # run by ´panel serve app.py´, then open http://localhost:5006/app in browser


# TODO: add params for matrix/pixelSize and BW like different vendors and handle their correlation
# TODO: add ACQ time
# TODO: add k-space plot
# TODO: add apodization
# TODO: parallel imaging (GRAPPA)
# TODO: B0 inhomogeneity
# TODO: Fast spin echo
# TODO: EPI