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
    'gray':       {'PD': 1.0, 'FF': .00, 'T1': {1.5: 1100, 3.0: 1330}, 'T2': {1.5:   92, 3.0:   80}, 'hexcolor': '00ff00'},
    'white':      {'PD': 0.9, 'FF': .00, 'T1': {1.5:  560, 3.0:  830}, 'T2': {1.5:   82, 3.0:  110}, 'hexcolor': 'd40000'},
    'CSF':        {'PD': 1.0, 'FF': .00, 'T1': {1.5: 4280, 3.0: 4160}, 'T2': {1.5: 2030, 3.0: 2100}, 'hexcolor': '00ffff'},
    'adipose':    {'PD': 1.0, 'FF': .95, 'T1': {1.5:  290, 3.0:  370}, 'T2': {1.5:  165, 3.0:  130}, 'hexcolor': 'ffe680'},
    'bonemarrow': {'PD': 1.0, 'FF': .50, 'T1': {1.5:  856, 3.0:  898}, 'T2': {1.5:   46, 3.0:   34}, 'hexcolor': 'ffff44'}, # relaxation times for water component
    'liver':      {'PD': 1.0, 'FF': .01, 'T1': {1.5:  586, 3.0:  809}, 'T2': {1.5:   46, 3.0:   34}, 'hexcolor': '800000'},
    'spleen':     {'PD': 1.0, 'FF': .00, 'T1': {1.5: 1057, 3.0: 1328}, 'T2': {1.5:   79, 3.0:   61}, 'hexcolor': 'ff0000'},
    'muscle':     {'PD': 1.0, 'FF': .00, 'T1': {1.5:  856, 3.0:  898}, 'T2': {1.5:   27, 3.0:   29}, 'hexcolor': '008000'},
    'kidneyMed':  {'PD': 1.0, 'FF': .00, 'T1': {1.5: 1412, 3.0: 1545}, 'T2': {1.5:   85, 3.0:   81}, 'hexcolor': 'aa4400'},
    'kidneyCor':  {'PD': 1.0, 'FF': .00, 'T1': {1.5:  966, 3.0: 1142}, 'T2': {1.5:   87, 3.0:   76}, 'hexcolor': '552200'},
    'spinalCord': {'PD': 1.0, 'FF': .00, 'T1': {1.5:  745, 3.0:  993}, 'T2': {1.5:   74, 3.0:   78}, 'hexcolor': 'ffff00'},
    'cortical':   {'PD': .05, 'FF': .00, 'T1': {1.5: 1000, 3.0: 1000}, 'T2': {1.5:    3, 3.0:    1}, 'hexcolor': '808000'},
    'blood':      {'PD': 1.0, 'FF': .00, 'T1': {1.5: 1441, 3.0: 1932}, 'T2': {1.5:  290, 3.0:  275}, 'hexcolor': 'ffffff'},
    'stomach':    {'PD': 1.0, 'FF': .00, 'T1': {1.5: 3000, 3.0: 3000}, 'T2': {1.5:  800, 3.0:  800}, 'hexcolor': '1a1a1a'},
    'perotineum': {'PD': 1.0, 'FF': .00, 'T1': {1.5: 1500, 3.0: 1500}, 'T2': {1.5:   30, 3.0:   30}, 'hexcolor': 'ff8080'},
}

FATRESONANCES = { 'Fat1':  {'shift': 0.9 - 4.7, 'ratio': .087, 'ratioWithFatSat': .010, 'PD': 1.0, 'T1': {1.5:  290, 3.0:  370}, 'T2': {1.5:  165, 3.0:  130}},
                  'Fat2':  {'shift': 1.3 - 4.7, 'ratio': .694, 'ratioWithFatSat': .033, 'PD': 1.0, 'T1': {1.5:  290, 3.0:  370}, 'T2': {1.5:  165, 3.0:  130}},
                  'Fat3':  {'shift': 2.1 - 4.7, 'ratio': .129, 'ratioWithFatSat': .038, 'PD': 1.0, 'T1': {1.5:  290, 3.0:  370}, 'T2': {1.5:  165, 3.0:  130}},
                  'Fat4':  {'shift': 2.8 - 4.7, 'ratio': .004, 'ratioWithFatSat': .003, 'PD': 1.0, 'T1': {1.5:  290, 3.0:  370}, 'T2': {1.5:  165, 3.0:  130}},
                  'Fat5':  {'shift': 4.3 - 4.7, 'ratio': .039, 'ratioWithFatSat': .037, 'PD': 1.0, 'T1': {1.5:  290, 3.0:  370}, 'T2': {1.5:  165, 3.0:  130}}, 
                  'Fat6':  {'shift': 5.3 - 4.7, 'ratio': .047, 'ratioWithFatSat': .045, 'PD': 1.0, 'T1': {1.5:  290, 3.0:  370}, 'T2': {1.5:  165, 3.0:  130}} }

SEQUENCES = ['Spin Echo', 'Spoiled Gradient Echo', 'Inversion Recovery']

DURATIONS = {'exc': 1.0, 'ref': 4.0, 'inv': 4.0, 'spoil': 1.0} # excitation, refocusing, inversion, spoiling [msec]

PHANTOMS = {
    'abdomen': {'FOV': (320, 400), 'matrix': (513, 641)} # odd matrix to ensure kspace center is sampled (not required)
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


def getKaxis(matrix, pixelSize, symmetric=True, fftshift=True):
    kax = np.fft.fftfreq(matrix)
    if symmetric and not (matrix%2): # half-pixel to make even matrix symmetric
        kax += 1/(2*matrix)
    kax /= pixelSize
    if fftshift:
        kax = np.fft.fftshift(kax)
    return kax


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
    kcenter = np.all(coords==0, axis=1)
    ksp[kcenter] = abs(sum([r[e-1,0]*r[e,1] - r[e,0]*r[e-1,1] for e in range(E)]))/2 # kspace center equals polygon area
    notkcenter = np.logical_not(kcenter)
    ksp[notkcenter] *= 1j / (2 * np.pi * np.linalg.norm(coords[notkcenter], axis=1)**2)
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


def getT2w(component, decayTime, dephasingTime, B0):
    T2 = TISSUES[component]['T2'][B0] if 'Fat' not in component else FATRESONANCES[component]['T2'][B0]
    T2prim = 100. # ad hoc value [msec]
    E2 = np.exp(-decayTime/T2)
    E2prim = np.exp(-np.abs(dephasingTime)/T2prim)
    return E2 * E2prim


def getPDandT1w(component, seqType, TR, TE, TI, FA, B0):
    PD = TISSUES[component]['PD'] if 'Fat' not in component else FATRESONANCES[component]['PD']
    T1 = TISSUES[component]['T1'][B0] if 'Fat' not in component else FATRESONANCES[component]['T1'][B0]

    E1 = np.exp(-TR/T1)
    if seqType == 'Spin Echo':
        return PD * (1 - E1)
    elif seqType == 'Spoiled Gradient Echo':
        return PD * np.sin(np.radians(FA)) * (1 - E1) / (1 - np.cos(np.radians(FA)) * E1)
    elif seqType == 'Inversion Recovery':
        return PD * (1 - 2 * np.exp(-TI/T1) + E1)
    else:
        raise Exception('Unknown sequence type: {}'.format(seqType))


TRvalues = [float('{:.2g}'.format(tr)) for tr in 10.**np.linspace(0, 4, 500)]
TEvalues = [float('{:.2g}'.format(te)) for te in 10.**np.linspace(0, 3, 500)]
TIvalues = [float('{:.2g}'.format(ti)) for ti in 10.**np.linspace(0, 4, 500)]


class MRIsimulator(param.Parameterized):
    object = param.ObjectSelector(default=list(PHANTOMS.keys())[0], objects=PHANTOMS.keys(), label='Phantom object')
    fieldStrength = param.ObjectSelector(default=1.5, objects=[1.5, 3.0], label='B0 field strength [T]')
    sequence = param.ObjectSelector(default=SEQUENCES[0], objects=SEQUENCES, label='Pulse sequence')
    FatSat = param.Boolean(default=False, label='Fat saturation')
    TR = param.Selector(default=8000, objects=TRvalues, label='TR [msec]')
    TE = param.Selector(default=7, objects=TEvalues, label='TE [msec]')
    FA = param.Number(default=90.0, bounds=(1, 90.0), precedence=-1, label='Flip angle [°]')
    TI = param.Selector(default=50, objects=TIvalues, precedence=-1, label='TI [msec]')
    FOVX = param.Number(default=420, bounds=(100, 600), label='FOV x [mm]')
    FOVY = param.Number(default=420, bounds=(100, 600), label='FOV y [mm]')
    matrixX = param.Integer(default=128, bounds=(16, 600), label='Acquisition matrix x')
    matrixY = param.Integer(default=128, bounds=(16, 600), label='Acquisition matrix y')
    reconMatrixX = param.Integer(default=256, bounds=(matrixX.default, 1024), label='Reconstruction matrix x')
    reconMatrixY = param.Integer(default=256, bounds=(matrixY.default, 1024), label='Reconstruction matrix y')
    frequencyDirection = param.ObjectSelector(default=list(DIRECTIONS.keys())[-1], objects=DIRECTIONS.keys(), label='Frequency encoding direction')
    pixelBandWidth = param.Number(default=2000, bounds=(50, 2000), label='Pixel bandwidth [Hz]')
    NSA = param.Integer(default=1, bounds=(1, 32), label='NSA')
    

    def __init__(self, **params):
        super().__init__(**params)
        self.updateReadoutDuration()
        self.updateTEbounds()
        self.updateTRbounds()

        self.fullPipeline = [   self.loadPhantom, 
                                self.sampleKspace, 
                                self.updateSamplingTime, 
                                self.modulateKspace, 
                                self.addNoise, 
                                self.updatePDandT1w, 
                                self.compileKspace, 
                                self.zerofill,
                                self.reconstruct]
        
        self.pipeline = set(self.fullPipeline)
    

    def runPipeline(self):
        for f in self.fullPipeline:
            if f in self.pipeline:
                f()
                self.pipeline.remove(f)
    

    def updateReadoutDuration(self):
        self.readoutDuration = 1e3 / self.pixelBandWidth # [msec]
        
    
    @param.depends('object', watch=True)
    def _watch_object(self):
        for f in self.fullPipeline:
            self.pipeline.add(f)
    

    @param.depends('FOVX', 'FOVY', watch=True)
    def _watch_FOV(self):
        for f in [self.sampleKspace, self.updateSamplingTime, self.modulateKspace, self.addNoise, self.compileKspace, self.zerofill, self.reconstruct]:
            self.pipeline.add(f)
    

    @param.depends('matrixX', 'matrixY', watch=True)
    def _watch_matrix(self):
        for f in [self.sampleKspace, self.updateSamplingTime, self.modulateKspace, self.addNoise, self.compileKspace, self.zerofill, self.reconstruct]:
            self.pipeline.add(f)
        self.param.reconMatrixX.bounds = (self.matrixX, self.param.reconMatrixX.bounds[1])
        self.reconMatrixX = max(self.reconMatrixX, self.matrixX)
        self.param.reconMatrixY.bounds = (self.matrixY, self.param.reconMatrixY.bounds[1])
        self.reconMatrixY = max(self.reconMatrixY, self.matrixY)


    @param.depends('frequencyDirection', watch=True)
    def _watch_frequencyDirection(self):
        for f in [self.sampleKspace, self.updateSamplingTime, self.modulateKspace, self.addNoise, self.compileKspace, self.zerofill, self.reconstruct]:
            self.pipeline.add(f)


    @param.depends('fieldStrength', watch=True)
    def _watch_fieldStrength(self):
        for f in [self.updateSamplingTime, self.modulateKspace, self.addNoise, self.updatePDandT1w, self.compileKspace, self.zerofill, self.reconstruct]:
            self.pipeline.add(f)
    

    @param.depends('pixelBandWidth', watch=True)
    def _watch_pixelBandWidth(self):
        for f in [self.updateSamplingTime, self.modulateKspace, self.addNoise, self.compileKspace, self.zerofill, self.reconstruct]:
            self.pipeline.add(f)
        self.updateReadoutDuration()


    @param.depends('NSA', watch=True)
    def _watch_NSA(self):
        for f in [self.updateSamplingTime, self.modulateKspace, self.addNoise, self.compileKspace, self.zerofill, self.reconstruct]:
            self.pipeline.add(f)


    @param.depends('sequence', watch=True)
    def _watch_sequence(self):
        for f in [self.modulateKspace, self.updatePDandT1w, self.compileKspace, self.zerofill, self.reconstruct]:
            self.pipeline.add(f)
        self.param.FA.precedence = 1 if self.sequence=='Spoiled Gradient Echo' else -1
        self.param.TI.precedence = 1 if self.sequence=='Inversion Recovery' else -1


    @param.depends('TE', watch=True)
    def _watch_TE(self):
        for f in [self.modulateKspace, self.updatePDandT1w, self.compileKspace, self.zerofill, self.reconstruct]:
            self.pipeline.add(f)
    

    @param.depends('TR', 'FA', 'TI', watch=True)
    def _watch_TR_FA_TI(self):
        for f in [self.updatePDandT1w, self.compileKspace, self.zerofill, self.reconstruct]:
            self.pipeline.add(f)
    
    
    @param.depends('FatSat', watch=True)
    def _watch_FatSat(self):
        for f in [self.compileKspace, self.zerofill, self.reconstruct]:
            self.pipeline.add(f)
    
    
    @param.depends('reconMatrixX', 'reconMatrixY', watch=True)
    def _watch_reconMatrix(self):
        for f in [self.zerofill, self.reconstruct]:
            self.pipeline.add(f)
    

    @param.depends('sequence', 'TE', 'TI', 'pixelBandWidth', watch=True)
    def updateTRbounds(self):
        minTR = DURATIONS['exc']/2 + self.TE + self.readoutDuration/2 + DURATIONS['spoil']
        if self.sequence == 'Inversion Recovery': minTR += DURATIONS['inv']/2 + self.TI - DURATIONS['exc']/2
        self.param.TR.objects = [tr for tr in TRvalues if not tr < minTR]
    
    
    @param.depends('sequence', 'TR', 'TI', 'pixelBandWidth', watch=True)
    def updateTEbounds(self):
        if self.sequence in ['Spin Echo', 'Inversion Recovery']: # seqs with refocusing pulse
            minTE = max((DURATIONS['exc'] + DURATIONS['ref']), (DURATIONS['ref'] + self.readoutDuration))
        else:
            minTE = (DURATIONS['exc'] + self.readoutDuration)/2
        maxTE = self.TR - (DURATIONS['exc'] + self.readoutDuration)/2 - DURATIONS['spoil']
        if self.sequence == 'Inversion Recovery':
            maxTE += (DURATIONS['exc'] - DURATIONS['inv'])/2 - self.TI
        self.param.TE.objects = [te for te in TEvalues if not te < minTE and not te > maxTE]
    

    @param.depends('sequence', 'TR', 'TE', 'pixelBandWidth', watch=True)
    def updateTIbounds(self):
        if self.sequence != 'Inversion Recovery': return
        minTI = (DURATIONS['inv'] + DURATIONS['exc'])/2
        maxTI = self.TR - (DURATIONS['inv']/2 + self.TE + self.readoutDuration/2 + DURATIONS['spoil'])
        self.param.TI.objects = [ti for ti in TIvalues if not ti < minTI and not ti > maxTI]
    

    @param.depends('sequence', 'TR', 'TE', 'TI', watch=True)
    def updateBWbounds(self):
        maxReadDurRightHalf = self.TR - self.TE - DURATIONS['exc']/2 - DURATIONS['spoil']
        if self.sequence == 'Inversion Recovery': maxReadDurRightHalf += (DURATIONS['exc'] - DURATIONS['inv'])/2 - self.TI
        if self.sequence in ['Spin Echo', 'Inversion Recovery']: # seqs with refocusing pulse
            maxReadDurLeftHalf = (self.TE - DURATIONS['ref'])/2
        else:
            maxReadDurLeftHalf = self.TE - DURATIONS['exc']/2
        maxReadDur = min(maxReadDurLeftHalf, maxReadDurRightHalf) * 2 * .99 # fudge factor
        minpBW = max(1e3 / maxReadDur, 50)
        self.param.pixelBandWidth.bounds = (minpBW, 2000)


    def loadPhantom(self):
        phantomPath = Path(__file__).parent.resolve() / 'phantoms/{p}'.format(p=self.object)
        self.phantom = PHANTOMS[self.object]
        self.phantom['kAxes'] = [getKaxis(self.phantom['matrix'][dim], self.phantom['FOV'][dim]/self.phantom['matrix'][dim]) for dim in range(len(self.phantom['matrix']))]

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
    

    def sampleKspace(self):
        self.matrix = [self.matrixY, self.matrixX]
        self.FOV = [self.FOVY, self.FOVX]
        self.freqDir = DIRECTIONS[self.frequencyDirection]
        
        self.oversampledMatrix = self.matrix.copy() # account for oversampling in frequency encoding direction
        if self.FOV[self.freqDir] < self.phantom['FOV'][self.freqDir]: # At least Nyquist sampling in frequency encoding direction
            self.oversampledMatrix[self.freqDir] = int(np.ceil(self.phantom['FOV'][self.freqDir] * self.matrix[self.freqDir] / self.FOV[self.freqDir]))
        
        self.kAxes = [getKaxis(self.oversampledMatrix[dim], self.FOV[dim]/self.matrix[dim]) for dim in range(len(self.matrix))]
        self.plainKspaceComps = resampleKspace(self.phantom, self.kAxes)
    

    def updateSamplingTime(self):
        self.samplingTime = self.kAxes[self.freqDir] * self.FOV[self.freqDir] / self.matrix[self.freqDir] / self.pixelBandWidth * 1e3 # msec
        self.noiseStd = 1. / np.sqrt(np.diff(self.samplingTime[:2]) * self.NSA) / self.fieldStrength
        self.samplingTime = np.expand_dims(self.samplingTime, axis=[dim for dim in range(len(self.matrix)) if dim != self.freqDir])
    

    def modulateKspace(self):
        decayTime = self.samplingTime + self.TE
        dephasingTime = decayTime if 'Gradient Echo' in self.sequence else self.samplingTime

        self.kspaceComps = {}
        for tissue in self.tissues:
            T2w = getT2w(tissue, decayTime, dephasingTime, self.fieldStrength)
            if TISSUES[tissue]['FF'] == .00:
                self.kspaceComps[tissue] = self.plainKspaceComps[tissue] * T2w
            else: # fat containing tissues
                self.kspaceComps[tissue + 'Water'] = self.plainKspaceComps[tissue] * T2w
                for component, resonance in FATRESONANCES.items():
                    T2w = getT2w(component, decayTime, dephasingTime, self.fieldStrength)
                    dephasing = np.exp(2j*np.pi * GYRO * self.fieldStrength * resonance['shift'] * dephasingTime * 1e-3)
                    self.kspaceComps[tissue + component] = self.plainKspaceComps[tissue] * dephasing * T2w
    
    
    def addNoise(self):
        self.noise = np.random.normal(0, self.noiseStd, self.oversampledMatrix) + 1j * np.random.normal(0, self.noiseStd, self.oversampledMatrix)


    def updatePDandT1w(self):
        self.PDandT1w = {component: getPDandT1w(component, self.sequence, self.TR, self.TE, self.TI, self.FA, self.fieldStrength) for component in self.tissues.union(set(FATRESONANCES.keys()))}


    def compileKspace(self):
        self.kspace = self.noise.copy()
        for component in self.kspaceComps:
            if 'Fat' in component:
                tissue = component[:component.find('Fat')]
                resonance = component[component.find('Fat'):]
                ratio = FATRESONANCES[resonance]['ratioWithFatSat' if self.FatSat else 'ratio']
                ratio *= TISSUES[tissue]['FF']
                self.kspace += self.kspaceComps[component] * self.PDandT1w[resonance] * ratio
            else:
                if 'Water' in component:
                    tissue = component[:component.find('Water')]
                    ratio = 1 - TISSUES[tissue]['FF']    
                else:
                    tissue = component
                    ratio = 1.0
                self.kspace += self.kspaceComps[component] * self.PDandT1w[tissue] * ratio

    
    def zerofill(self):
        self.reconMatrix = [self.reconMatrixY, self.reconMatrixX]
        self.oversampledReconMatrix = self.reconMatrix.copy()
        if self.FOV[self.freqDir] < self.phantom['FOV'][self.freqDir]: # At least Nyquist sampling in frequency encoding direction
            self.oversampledReconMatrix[self.freqDir] = int(self.reconMatrix[self.freqDir] * self.oversampledMatrix[self.freqDir] / self.matrix[self.freqDir])
        self.zerofilledkspace = zerofill(np.fft.ifftshift(self.kspace), self.oversampledReconMatrix)
    
    
    def reconstruct(self):
        pixelShifts = [.0] * len(self.reconMatrix) # pixel shift
        for dim in range(len(pixelShifts)):
            if not self.oversampledReconMatrix[dim]%2:
                pixelShifts[dim] += 1/2 # half pixel shift for even matrixsize due to fft
            if (self.oversampledReconMatrix[dim] - self.reconMatrix[dim])%2:
                pixelShifts[dim] += 1/2 # half pixel shift due to cropping an odd number of pixels in image space
        halfPixelShift = getPixelShiftMatrix(self.oversampledReconMatrix, pixelShifts)
        sampleShifts = [0. if self.matrix[dim]%2 else .5 for dim in range(len(self.matrix))]
        halfSampleShift = getPixelShiftMatrix(self.oversampledReconMatrix, sampleShifts)
        
        kspace = self.zerofilledkspace * halfPixelShift
        self.imageArray = np.fft.ifft2(kspace) * halfSampleShift
        self.imageArray = crop(np.fft.fftshift(self.imageArray), self.reconMatrix)
    
    
    @param.depends('fieldStrength', 'sequence', 'FatSat', 'TR', 'TE', 'FA', 'TI', 'FOVX', 'FOVY', 'matrixX', 'matrixY', 'reconMatrixX', 'reconMatrixY', 'frequencyDirection', 'pixelBandWidth', 'NSA')
    def getKspace(self):
        self.runPipeline()
        kAxes = []
        for dim in range(2):
            kAxes.append(getKaxis(self.oversampledReconMatrix[dim], self.FOV[dim]/self.reconMatrix[dim]))
            # half-sample shift axis when odd number of zeroes:
            if (self.oversampledReconMatrix[dim]-self.oversampledMatrix[dim])%2:
                shift = self.reconMatrix[dim] / (2 * self.oversampledReconMatrix[dim] * self.FOV[dim])
                kAxes[-1] += shift * (-1)**(self.matrix[dim]%2)
        ksp = xr.DataArray(
            np.abs(np.fft.fftshift(self.zerofilledkspace))**.2, 
            dims=('ky', 'kx'),
            coords={'kx': kAxes[1], 'ky': kAxes[0][::-1]}
        )
        ksp.kx.attrs['units'] = ksp.ky.attrs['units'] = '1/mm'
        return hv.Image(ksp, vdims=['magnitude']).options(cmap='gray', aspect='equal')
    

    @param.depends('fieldStrength', 'sequence', 'FatSat', 'TR', 'TE', 'FA', 'TI', 'FOVX', 'FOVY', 'matrixX', 'matrixY', 'reconMatrixX', 'reconMatrixY', 'frequencyDirection', 'pixelBandWidth', 'NSA')
    def getImage(self):
        self.runPipeline()
        iAxes = [(np.arange(self.reconMatrix[dim]) - (self.reconMatrix[dim]-1)/2) / self.reconMatrix[dim] * self.FOV[dim] for dim in range(2)]
        img = xr.DataArray(
            np.abs(self.imageArray), 
            dims=('y', 'x'),
            coords={'x': iAxes[1], 'y': iAxes[0][::-1]}
        )
        img.x.attrs['units'] = img.y.attrs['units'] = 'mm'
        return hv.Image(img, vdims=['magnitude']).options(cmap='gray', aspect='equal')


explorer = MRIsimulator(name='')
title = '# SpinSight MRI simulator'
author = '*Written by [Johan Berglund](mailto:johan.berglund@akademiska.se), Ph.D.*'
contrastParams = pn.panel(explorer.param, parameters=['fieldStrength', 'sequence', 'FatSat', 'TR', 'TE', 'FA', 'TI'], widgets={'TR': pn.widgets.DiscreteSlider, 'TE': pn.widgets.DiscreteSlider, 'TI': pn.widgets.DiscreteSlider}, name='Contrast')
geometryParams = pn.panel(explorer.param, parameters=['FOVX', 'FOVY', 'matrixX', 'matrixY', 'reconMatrixX', 'reconMatrixY', 'frequencyDirection', 'pixelBandWidth', 'NSA'], name='Geometry')
dmapKspace = hv.DynamicMap(explorer.getKspace).opts(frame_height=500)
dmapMRimage = hv.DynamicMap(explorer.getImage).opts(frame_height=500)
dashboard = pn.Row(pn.Column(pn.pane.Markdown(title), pn.Row(contrastParams, geometryParams), pn.pane.Markdown(author)), pn.Column(dmapMRimage, dmapKspace))
dashboard.servable() # run by ´panel serve app.py´, then open http://localhost:5006/app in browser

# TODO: bug when switching sequence and pulses are tight
# TODO: phase oversampling
# TODO: abdomen phantom ribs, pancreas, hepatic arteries
# TODO: add params for matrix/pixelSize and BW like different vendors and handle their correlation
# TODO: add ACQ time
# TODO: add apodization
# TODO: parallel imaging (GRAPPA)
# TODO: B0 inhomogeneity
# TODO: Fast spin echo
# TODO: EPI