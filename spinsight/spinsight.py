import holoviews as hv
import panel as pn
import param
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import re
import xarray as xr
import sequence
from bokeh.models import HoverTool
from functools import partial

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

PHANTOMS = {
    'abdomen': {'FOV': (320, 400), 'matrix': (513, 641)}, # odd matrix to ensure kspace center is sampled (not required)
    'brain': {'FOV': (188, 156), 'matrix': (601, 601)} # odd matrix to ensure kspace center is sampled (not required)
}

DIRECTIONS = {'anterior-posterior': 0, 'left-right': 1}


def isGradientEcho(sequence):
    return 'Gradient Echo' in sequence


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
        if hexcolor not in [v['hexcolor'] for v in TISSUES.values()]:
            raise Exception('No tissue corresponding to hexcolor {}'.format(hexcolor))
        tissue = [tissue for tissue in TISSUES if TISSUES[tissue]['hexcolor']==hexcolor][0]
        translation, rotation, scale = parseTransform(path.attrib['transform'] if 'transform' in path.attrib else '')
        if rotation != 0 or translation != (0, 0):
            raise NotImplementedError()
        subpaths = getSubpaths(path.attrib['d'], scale)
        for subpath in subpaths:
            polygons.append({'vertices': subpath, 'tissue': tissue})
    return polygons


def getKaxis(matrix, pixelSize, symmetric=True, fftshift=True):
    kax = np.fft.fftfreq(matrix)
    if symmetric and not (matrix%2): # half-pixel to make even matrix symmetric
        kax += 1/(2*matrix)
    kax /= pixelSize
    if fftshift:
        kax = np.fft.fftshift(kax)
    return kax


def kspacePolygon(poly, k):
    # analytical 2D Fourier transform of polygon (see https://cvil.ucsd.edu/wp-content/uploads/2016/09/Realistic-analytical-polyhedral-MRI-phantoms.pdf)
    r = poly['vertices'] # position vectors of vertices Ve
    Lv = np.roll(r, -1, axis=1) - r # edge vectors
    L = np.linalg.norm(Lv, axis=0) # edge lengths
    t = Lv/L # edge unit vectors
    n = np.array([-t[1,:], t[0,:]]) # normals to tangents (pointing out from polygon)
    rc = r + Lv / 2 # position vector for center of edge

    ksp = np.sum(L * np.dot(k, n) * np.sinc(np.dot(k, Lv)) * np.exp(-2j*np.pi * np.dot(k, rc)), axis=-1)
    
    kcenter = np.all(k==0, axis=-1)
    ksp[kcenter] = polygonArea(r)
    notkcenter = np.logical_not(kcenter)
    ksp[notkcenter] *= 1j / (2 * np.pi * np.linalg.norm(k[notkcenter], axis=-1)**2)
    return ksp


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


def updateBounds(curval, values, minval=None, maxval=None):
        if minval is not None:
            values = [val for val in values if not val < minval]
        if maxval is not None:
            values = [val for val in values if not val > maxval]
        value = min(values, key=lambda x: abs(x-curval))
        return values, value


def bounds_hook(plot, elem, xbounds=None):
    x_range = plot.handles['plot'].x_range
    if xbounds is not None:
        x_range.bounds = xbounds
    else:
        x_range.bounds = x_range.start, x_range.end 


def hideframe_hook(plot, elem):
    plot.handles['plot'].outline_line_color = None


TRvalues = [float('{:.2g}'.format(tr)) for tr in 10.**np.linspace(0, 4, 500)]
TEvalues = [float('{:.2g}'.format(te)) for te in 10.**np.linspace(0, 3, 500)]
TIvalues = [float('{:.2g}'.format(ti)) for ti in 10.**np.linspace(0, 4, 500)]


class MRIsimulator(param.Parameterized):
    object = param.ObjectSelector(default='brain', objects=PHANTOMS.keys(), label='Phantom object')
    fieldStrength = param.ObjectSelector(default=1.5, objects=[1.5, 3.0], label='B0 field strength [T]')
    sequence = param.ObjectSelector(default=SEQUENCES[0], objects=SEQUENCES, label='Pulse sequence')
    FatSat = param.Boolean(default=False, label='Fat saturation')
    TR = param.Selector(default=8000, objects=TRvalues, label='TR [msec]')
    TE = param.Selector(default=7, objects=TEvalues, label='TE [msec]')
    FA = param.Number(default=90.0, bounds=(1, 90.0), precedence=-1, label='Flip angle [°]')
    TI = param.Selector(default=50, objects=TIvalues, precedence=-1, label='TI [msec]')
    FOVF = param.Number(default=420, bounds=(100, 600), label='FOV x [mm]')
    FOVP = param.Number(default=420, bounds=(100, 600), label='FOV y [mm]')
    matrixF = param.Integer(default=128, bounds=(16, 600), label='Acquisition matrix x')
    matrixP = param.Integer(default=128, bounds=(16, 600), label='Acquisition matrix y')
    reconMatrixF = param.Integer(default=256, bounds=(matrixF.default, 1024), label='Reconstruction matrix x')
    reconMatrixP = param.Integer(default=256, bounds=(matrixP.default, 1024), label='Reconstruction matrix y')
    frequencyDirection = param.ObjectSelector(default=list(DIRECTIONS.keys())[-1], objects=DIRECTIONS.keys(), label='Frequency encoding direction')
    pixelBandWidth = param.Number(default=2000, bounds=(125, 2000), label='Pixel bandwidth [Hz]')
    NSA = param.Integer(default=1, bounds=(1, 32), label='NSA')
    

    def __init__(self, **params):
        super().__init__(**params)

        self.fullReconPipeline = [
            self.loadPhantom, 
            self.sampleKspace, 
            self.updateSamplingTime, 
            self.modulateKspace, 
            self.addNoise, 
            self.updatePDandT1w, 
            self.compileKspace, 
            self.zerofill, 
            self.reconstruct
        ]
        
        self.reconPipeline = set(self.fullReconPipeline)

        self.timeDim = hv.Dimension('time', label='time', unit='ms')

        self.boards = { 'frequency': {'dim': hv.Dimension('frequency', label='G read', unit='mT/m', range=(-30, 30)), 'color': 'cadetblue'}, 
                        'phase': {'dim': hv.Dimension('phase', label='G phase', unit='mT/m', range=(-30, 30)), 'color': 'cadetblue'}, 
                        'slice': {'dim': hv.Dimension('slice', label='G slice', unit='mT/m', range=(-30, 30)), 'color': 'cadetblue'}, 
                        'RF': {'dim': hv.Dimension('RF', label='RF', unit='μT', range=(-5, 25)), 'color': 'red'},
                        'ADC': {'dim': hv.Dimension('ADC', label='ADC', unit=''), 'color': 'orange'} }

        self._watch_reconMatrix()

        self.boardPlots = {board: {'hline': hv.HLine(0.0, kdims=[self.timeDim, self.boards[board]['dim']]).opts(tools=['xwheel_zoom', 'xpan', 'reset'], default_tools=[], active_tools=['xwheel_zoom', 'xpan'])} for board in self.boards if board != 'ADC'}

        hv.opts.defaults(hv.opts.Image(width=500, height=500, invert_yaxis=False, toolbar='below', cmap='gray', aspect='equal'))
        hv.opts.defaults(hv.opts.HLine(line_width=1.5, line_color='gray'))
        hv.opts.defaults(hv.opts.VSpan(color='orange', fill_alpha=.1, hover_fill_alpha=.8, default_tools=[]))
        hv.opts.defaults(hv.opts.Overlay(width=1700, height=120, border=4, show_grid=False, xaxis=None))
        hv.opts.defaults(hv.opts.Area(fill_alpha=.5, line_width=1.5, line_color='gray', default_tools=[]))
        hv.opts.defaults(hv.opts.Polygons(line_width=1.5, fill_alpha=0, line_alpha=0, line_color='gray', selection_line_color='black', hover_fill_alpha=.8, hover_line_alpha=1, selection_fill_alpha=.8, selection_line_alpha=1, nonselection_line_alpha=0, default_tools=[]))

        self.maxAmp = 25. # mT/m
        self.maxSlew = 80. # T/m/s

        for board in self.boards:
            self.boards[board]['objects'] = {}

        self.fullSequencePipeline = [
            self.setupExcitation, 
            self.setupRefocusing,
            self.setupInversion,
            self.setupSliceSelection,
            self.setupReadout,
            self.setupPhaser,
            self.setupSpoiler,
            self.placeRefocusing,
            self.placeInversion,
            self.placeReadout,
            self.placePhaser,
            self.placeSpoiler,
            self.renderFrequencyBoard, 
            self.renderPhaseBoard, 
            self.renderSliceBoard, 
            self.renderRFBoard,
            self.updateMinTE,
            self.updateMinTI,
            self.updateMinTR,
            self.updateMaxTE,
            self.updateMaxTI,
            self.updateBWbounds,
            self.updateMatrixFBounds,
            self.updateFOVFbounds
        ]

        self.sequencePipeline = set(self.fullSequencePipeline)
        self.runSequencePipeline()


    def runReconPipeline(self):
        for f in self.fullReconPipeline:
            if f in self.reconPipeline:
                f()
                self.reconPipeline.remove(f)


    def runSequencePipeline(self):
        for f in self.fullSequencePipeline:
            if f in self.sequencePipeline:
                f()
                self.sequencePipeline.remove(f)
    
    
    @param.depends('object', watch=True)
    def _watch_object(self):
        for f in self.fullReconPipeline:
            self.reconPipeline.add(f)
    

    @param.depends('FOVF', watch=True)
    def _watch_FOVF(self):
        for f in [self.sampleKspace, self.updateSamplingTime, self.modulateKspace, self.addNoise, self.compileKspace, self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)
        for f in [self.setupReadout, self.updateBWbounds, self.updateMatrixFBounds]:
            self.sequencePipeline.add(f)
        
    

    @param.depends('FOVP', watch=True)
    def _watch_FOVP(self):
        for f in [self.sampleKspace, self.updateSamplingTime, self.modulateKspace, self.addNoise, self.compileKspace, self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)
        self.sequencePipeline.add(self.setupPhaser)
    
    
    @param.depends('matrixF', watch=True)
    def _watch_matrixF(self):
        for f in [self.sampleKspace, self.updateSamplingTime, self.modulateKspace, self.addNoise, self.compileKspace, self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)
        for f in [self.setupReadout, self.updateBWbounds, self.updateFOVFbounds]:
            self.sequencePipeline.add(f)
        self.param.reconMatrixF.bounds = (self.matrixF, self.param.reconMatrixF.bounds[1])
        self.reconMatrixF = min(max(int(self.matrixF * self.recAcqRatioF), self.matrixF), self.param.reconMatrixF.bounds[1])
    
    
    @param.depends('matrixP', watch=True)
    def _watch_matrixP(self):
        for f in [self.sampleKspace, self.updateSamplingTime, self.modulateKspace, self.addNoise, self.compileKspace, self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)
        self.sequencePipeline.add(self.setupPhaser)
        self.param.reconMatrixP.bounds = (self.matrixP, self.param.reconMatrixP.bounds[1])
        self.reconMatrixP = min(max(int(self.matrixP * self.recAcqRatioP), self.matrixP), self.param.reconMatrixP.bounds[1])


    @param.depends('frequencyDirection', watch=True)
    def _watch_frequencyDirection(self):
        for f in [self.sampleKspace, self.updateSamplingTime, self.modulateKspace, self.addNoise, self.compileKspace, self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)
        for p in [self.param.FOVF, self.param.FOVP, self.param.matrixF, self.param.matrixP, self.param.reconMatrixF, self.param.reconMatrixP]:
            if ' x' in p.label:
                p.label = p.label.replace(' x', ' y')
            elif ' y' in p.label:
                p.label = p.label.replace(' y', ' x')


    @param.depends('fieldStrength', watch=True)
    def _watch_fieldStrength(self):
        for f in [self.updateSamplingTime, self.modulateKspace, self.addNoise, self.updatePDandT1w, self.compileKspace, self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)
    

    @param.depends('pixelBandWidth', watch=True)
    def _watch_pixelBandWidth(self):
        for f in [self.updateSamplingTime, self.modulateKspace, self.addNoise, self.compileKspace, self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)
        for f in [self.setupReadout, self.updateMatrixFBounds, self.updateFOVFbounds]:
            self.sequencePipeline.add(f)


    @param.depends('NSA', watch=True)
    def _watch_NSA(self):
        for f in [self.updateSamplingTime, self.modulateKspace, self.addNoise, self.compileKspace, self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)


    @param.depends('sequence', watch=True)
    def _watch_sequence(self):
        for f in [self.modulateKspace, self.updatePDandT1w, self.compileKspace, self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)
        for f in [self.setupExcitation, self.setupRefocusing, self.setupInversion, self.placeReadout, self.placePhaser]:
            self.sequencePipeline.add(f)
        self.param.FA.precedence = 1 if self.sequence=='Spoiled Gradient Echo' else -1
        self.param.TI.precedence = 1 if self.sequence=='Inversion Recovery' else -1
        tr = self.TR
        self.TR = self.param.TR.objects[-1] # max TR
        self.runSequencePipeline
        self.TE = min(self.param.TE.objects, key=lambda x: abs(x-self.TE)) # TE within bounds
        self.runSequencePipeline
        if self.sequence=='Inversion Recovery':
            self.TI = min(self.param.TI.objects, key=lambda x: abs(x-self.TI)) # TI within bounds
            self.runSequencePipeline
        self.TR = min(self.param.TR.objects, key=lambda x: abs(x-tr)) # Set back TR within bounds
        self.runSequencePipeline
    

    @param.depends('TE', watch=True)
    def _watch_TE(self):
        for f in [self.modulateKspace, self.updatePDandT1w, self.compileKspace, self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)
        for f in [self.placeRefocusing, self.placeReadout, self.placePhaser, self.updateMatrixFBounds, self.updateFOVFbounds]:
            self.sequencePipeline.add(f)
    

    @param.depends('TR', watch=True)
    def _watch_TR(self):
        for f in [self.updatePDandT1w, self.compileKspace, self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)
        for f in [self.updateMaxTE, self.updateMaxTI, self.updateBWbounds, self.renderFrequencyBoard, self.renderPhaseBoard, self.renderSliceBoard, self.renderRFBoard]:
            self.sequencePipeline.add(f)
    

    @param.depends('TI', watch=True)
    def _watch_TI(self):
        for f in [self.updatePDandT1w, self.compileKspace, self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)
        self.sequencePipeline.add(self.placeInversion)


    @param.depends('FA', watch=True)
    def _watch_FA(self):
        for f in [self.updatePDandT1w, self.compileKspace, self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)
        self.sequencePipeline.add(self.setupExcitation)
    
    
    @param.depends('FatSat', watch=True)
    def _watch_FatSat(self):
        for f in [self.compileKspace, self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)
        # TODO: add fatsat sequence objects!
    
    
    @param.depends('reconMatrixF', 'reconMatrixP', watch=True)
    def _watch_reconMatrix(self):
        self.recAcqRatioF = self.reconMatrixF / self.matrixF
        self.recAcqRatioP = self.reconMatrixP / self.matrixP
        for f in [self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)
    
    
    def getSeqStart(self):
        if self.sequence == 'Inversion Recovery': 
            return self.boards['slice']['objects']['slice select inversion']['time'][0]
        else:
            return self.boards['slice']['objects']['slice select excitation']['time'][0]
    
    
    def updateMinTE(self):
        if not isGradientEcho(self.sequence):
            leftSide = max(
                self.boards['frequency']['objects']['read prephaser']['time'][-1], 
                self.boards['slice']['objects']['slice select rephaser']['time'][-1] + (self.boards['slice']['objects']['slice select refocusing']['riseTime_f']))
            leftSide += self.boards['RF']['objects']['refocusing']['dur_f'] / 2
            rightSide = max(
                self.boards['frequency']['objects']['readout']['riseTime_f'],
                self.boards['phase']['objects']['phase encode']['dur_f'],
                self.boards['slice']['objects']['slice select refocusing']['riseTime_f'])
            rightSide += (self.boards['RF']['objects']['refocusing']['dur_f'] + self.boards['ADC']['objects']['sampling']['dur_f']) / 2
            self.minTE = max(leftSide, rightSide) * 2
        else:
            self.minTE = max(
                self.boards['frequency']['objects']['read prephaser']['dur_f'] + self.boards['frequency']['objects']['readout']['riseTime_f'],
                self.boards['phase']['objects']['phase encode']['dur_f'],
                self.boards['slice']['objects']['slice select excitation']['riseTime_f'] + self.boards['slice']['objects']['slice select rephaser']['dur_f']
            )
            self.minTE += (self.boards['RF']['objects']['excitation']['dur_f'] + self.boards['ADC']['objects']['sampling']['dur_f']) / 2
        self.sequencePipeline.add(self.updateMaxTE)


    def updateMinTI(self):
        if self.sequence != 'Inversion Recovery': return
        self.minTI = sum([
            self.boards['RF']['objects']['inversion']['dur_f'] / 2, 
            self.boards['slice']['objects']['inversion spoiler']['dur_f'], 
            self.boards['RF']['objects']['excitation']['dur_f'] / 2 ])
        self.sequencePipeline.add(self.updateMaxTI)
    
    
    def updateMinTR(self):
        self.minTR = self.boards['slice']['objects']['spoiler']['time'][-1]
        self.minTR -= self.getSeqStart()
        self.param.TR.objects, _ = updateBounds(self.TR, TRvalues, minval=self.minTR)
        self.sequencePipeline.add(self.updateMaxTE)
        self.sequencePipeline.add(self.updateMaxTI)
    

    def updateMaxTE(self):
        maxTE = self.TR - self.minTR + self.TE
        self.param.TE.objects, _ = updateBounds(self.TE, TEvalues, minval=self.minTE, maxval=maxTE)
    
    
    def updateMaxTI(self):
        if self.sequence != 'Inversion Recovery': return
        maxTI = self.TR - self.minTR + self.TI
        self.param.TI.objects, _ = updateBounds(self.TI, TIvalues, minval=self.minTI, maxval=maxTI)
    
    
    def getMaxPrephaserArea(self):
        maxRiseTime = self.maxAmp/self.maxSlew
        if isGradientEcho(self.sequence):
            maxPrephaserDur =  self.boards['ADC']['objects']['sampling']['time'][0] - self.boards['RF']['objects']['excitation']['time'][-1] - maxRiseTime # use max risetime to be on the safe side
        else:
            maxPrephaserDur =  self.boards['RF']['objects']['refocusing']['time'][0] - self.boards['RF']['objects']['excitation']['time'][-1]
        maxPrephaserFlatDur = maxPrephaserDur - (2 * maxRiseTime)
        if maxPrephaserFlatDur < 0: # triangle
            maxPrephaserArea = maxPrephaserDur**2 * self.maxSlew / 4
        else: # trapezoid
            slewArea = self.maxAmp**2 / self.maxSlew
            flatArea = self.maxAmp * maxPrephaserFlatDur
            maxPrephaserArea = slewArea + flatArea
        return maxPrephaserArea
    

    def getMaxReadoutArea(self):
        # max wrt prephaser (use maxAmp to be on the safe side)
        maxReadoutArea1 = self.getMaxPrephaserArea() * 2 - self.maxAmp**2 / self.maxSlew
        maxReadoutArea2 = self.maxAmp * 1e3 / self.pixelBandWidth # max wrt maxAmp
        return min(maxReadoutArea1, maxReadoutArea2)
    

    def updateBWbounds(self):
        freeSpaceRight = self.TR - (self.TE - self.getSeqStart()) - self.boards['slice']['objects']['spoiler']['dur_f']
        if isGradientEcho(self.sequence):
            freeSpaceLeft = self.TE - self.boards['RF']['objects']['excitation']['time'][-1]
            freeSpaceLeft -= max(
                self.boards['frequency']['objects']['read prephaser']['dur_f'] + self.boards['frequency']['objects']['readout']['riseTime_f'],
                self.boards['phase']['objects']['phase encode']['dur_f'],
                self.boards['slice']['objects']['slice select excitation']['riseTime_f'] + self.boards['slice']['objects']['slice select rephaser']['dur_f'])
        else:
            freeSpaceLeft = self.TE - self.boards['RF']['objects']['refocusing']['time'][-1]
            freeSpaceLeft -= max(
                self.boards['frequency']['objects']['readout']['riseTime_f'],
                self.boards['phase']['objects']['phase encode']['dur_f'],
                self.boards['slice']['objects']['slice select refocusing']['riseTime_f'])
        maxReadDur = min(freeSpaceLeft, freeSpaceRight) * 2 * .99
        minpBW = max(1e3 / maxReadDur, 125)
        readoutArea = 1e3 * self.matrixF / (self.FOVF * GYRO)
        BWlimit = 1e3 * self.maxAmp / readoutArea * .99
        maxpBW = min(BWlimit, 2000)
        ampLimit = np.sqrt((self.getMaxPrephaserArea() * 2 - readoutArea) * self.maxSlew)
        BWlimit = 1e3 * ampLimit / readoutArea * .99
        maxpBW = min(BWlimit, maxpBW)
        self.param.pixelBandWidth.bounds = (minpBW, maxpBW)
    

    def updateMatrixFBounds(self):
        maxMatrixF = int(self.getMaxReadoutArea() * 1e-3 * (self.FOVF * GYRO))
        self.param.matrixF.bounds = (16, min(maxMatrixF, 600))

    
    def updateFOVFbounds(self):
        minFOVF = 1e3 * self.matrixF / (self.getMaxReadoutArea() * GYRO) * 1.01
        self.param.FOVF.bounds = (max(minFOVF, 100), 600)
    

    def loadPhantom(self):
        phantomPath = Path(__file__).parent.parent.resolve() / 'phantoms/{p}'.format(p=self.object)
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
            print('Preparing k-space for "{}" phantom. This might take a few minutes on first use...'.format(self.object), end='', flush=True)
            polys = readSVG(Path(phantomPath / self.object).with_suffix('.svg'))
            self.tissues = set([poly['tissue'] for poly in polys])
            self.phantom['kspace'] = {tissue: np.zeros((self.phantom['matrix']), dtype=complex) for tissue in self.tissues}
            k = np.array(np.meshgrid(self.phantom['kAxes'][0], self.phantom['kAxes'][1])).T
            for poly in polys:
                self.phantom['kspace'][poly['tissue']] += kspacePolygon(poly, k)
            for tissue in self.tissues:
                file = Path(phantomPath / tissue).with_suffix('.npy')
                np.save(file, self.phantom['kspace'][tissue])
            print('DONE')
    

    def sampleKspace(self):
        self.matrix = [self.matrixP, self.matrixF]
        self.FOV = [self.FOVP, self.FOVF]
        self.freqDir = DIRECTIONS[self.frequencyDirection]
        if self.freqDir==0:
            self.matrix.reverse()
            self.FOV.reverse()
        
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
        dephasingTime = decayTime if isGradientEcho(self.sequence) else self.samplingTime

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
        self.reconMatrix = [self.reconMatrixP, self.reconMatrixF]
        if self.freqDir==0:
            self.reconMatrix.reverse()
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
    
    
    def setupExcitation(self):
        FA = self.FA if isGradientEcho(self.sequence) else 90.
        self.boards['RF']['objects']['excitation'] = sequence.getRF(flipAngle=FA, time=0., dur=2., shape='hammingSinc',  name='excitation')
        for f in [self.setupSliceSelection, self.renderRFBoard, self.updateMinTE, self.updateMinTI, self.updateBWbounds, self.updateMatrixFBounds, self.updateFOVFbounds]:
            self.sequencePipeline.add(f)

    def setupRefocusing(self):
        if not isGradientEcho(self.sequence):
            self.boards['RF']['objects']['refocusing'] = sequence.getRF(flipAngle=180., dur=3., shape='hammingSinc',  name='refocusing')
            self.sequencePipeline.add(self.placeRefocusing)
        else:
            if 'refocusing' in self.boards['RF']['objects']:
                del self.boards['RF']['objects']['refocusing']
        self.sequencePipeline.add(self.setupSliceSelection)
        self.sequencePipeline.add(self.renderRFBoard)
        self.sequencePipeline.add(self.updateMinTE)


    def setupInversion(self):
        if self.sequence=='Inversion Recovery':
            self.boards['RF']['objects']['inversion'] = sequence.getRF(flipAngle=180., dur=3., shape='hammingSinc',  name='inversion')
            self.sequencePipeline.add(self.placeInversion)
        else:
            if 'inversion' in self.boards['RF']['objects']:
                del self.boards['RF']['objects']['inversion']
        self.sequencePipeline.add(self.setupSliceSelection)
        self.sequencePipeline.add(self.renderRFBoard)
        self.sequencePipeline.add(self.updateMinTI)
    

    def setupSliceSelection(self):
        flatDur = self.boards['RF']['objects']['excitation']['dur_f']
        sliceSelectAmp = 10. # TODO: determine from slice thickness
        sliceSelectExcitation = sequence.getGradient('slice', 0., maxAmp=sliceSelectAmp, flatDur=flatDur, name='slice select excitation')
        sliceRephaserArea = -sliceSelectExcitation['area_f']/2
        sliceSelectRephaser = sequence.getGradient('slice', totalArea=sliceRephaserArea, name='slice select rephaser')
        rephaserTime = (sliceSelectExcitation['dur_f'] + sliceSelectRephaser['dur_f']) / 2
        sequence.moveWaveform(sliceSelectRephaser, rephaserTime)
        self.boards['slice']['objects']['slice select excitation'] = sliceSelectExcitation
        self.boards['slice']['objects']['slice select rephaser'] = sliceSelectRephaser

        if 'refocusing' in self.boards['RF']['objects']:
            flatDur = self.boards['RF']['objects']['refocusing']['dur_f']
            self.boards['slice']['objects']['slice select refocusing'] = sequence.getGradient('slice', maxAmp=sliceSelectAmp, flatDur=flatDur, name='slice select refocusing')
            self.sequencePipeline.add(self.placeRefocusing)
        elif 'slice select refocusing' in self.boards['slice']['objects']:
            del self.boards['slice']['objects']['slice select refocusing']
            
        if 'inversion' in self.boards['RF']['objects']:
            flatDur = self.boards['RF']['objects']['inversion']['dur_f']
            self.boards['slice']['objects']['slice select inversion'] = sequence.getGradient('slice', maxAmp=sliceSelectAmp, flatDur=flatDur, name='slice select inversion')

            spoilerArea = 30. # uTs/m
            self.boards['slice']['objects']['inversion spoiler'] = sequence.getGradient('slice', totalArea=spoilerArea, name='inversion spoiler')
            self.sequencePipeline.add(self.placeInversion)
        elif 'slice select inversion' in self.boards['slice']['objects']:
            del self.boards['slice']['objects']['slice select inversion']
            del self.boards['slice']['objects']['inversion spoiler']
        
        self.sequencePipeline.add(self.renderFrequencyBoard) # due to TR bounds
        self.sequencePipeline.add(self.renderPhaseBoard) # due to TR bounds
        self.sequencePipeline.add(self.renderSliceBoard)
        self.sequencePipeline.add(self.renderRFBoard) # due to TR bounds
        self.sequencePipeline.add(self.updateMinTE)
        self.sequencePipeline.add(self.updateMinTI)
        self.sequencePipeline.add(self.updateMinTR)
        self.sequencePipeline.add(self.updateBWbounds)
    

    def setupReadout(self):
        flatArea = self.matrixF / (self.FOVF/1e3 * GYRO) # uTs/m
        amp = self.pixelBandWidth * self.matrixF / (self.FOVF * GYRO) # mT/m
        readout = sequence.getGradient('frequency', maxAmp=amp, flatArea=flatArea, name='readout')
        self.boards['ADC']['objects']['sampling'] = sequence.getADC(dur=readout['flatDur_f'], name='sampling')
        prephaser = sequence.getGradient('frequency', totalArea=readout['area_f']/2, name='read prephaser')        
        self.boards['frequency']['objects']['readout'] = readout
        self.boards['frequency']['objects']['read prephaser'] = prephaser
        self.sequencePipeline.add(self.placeReadout)
        self.sequencePipeline.add(self.updateMinTE)
    
    
    def setupPhaser(self):
        maxArea = self.matrixP / (self.FOVP/1e3 * GYRO * 2) # uTs/m
        self.boards['phase']['objects']['phase encode'] = sequence.getGradient('phase', totalArea=maxArea, name='phase encode')
        self.sequencePipeline.add(self.placePhaser)
        self.sequencePipeline.add(self.updateMinTE)
        self.sequencePipeline.add(self.updateBWbounds)
    

    def setupSpoiler(self):
        spoilerArea = 30. # uTs/m
        self.boards['slice']['objects']['spoiler'] = sequence.getGradient('slice', totalArea=spoilerArea, name='spoiler')
        self.sequencePipeline.add(self.placeSpoiler)
    
    
    def placeRefocusing(self):
        for board, name, renderer in [('RF', 'refocusing', self.renderRFBoard), ('slice', 'slice select refocusing', self.renderSliceBoard)]:
            if name in self.boards[board]['objects']:
                sequence.moveWaveform(self.boards[board]['objects'][name], self.TE/2)
                self.sequencePipeline.add(renderer)
        self.sequencePipeline.add(self.updateBWbounds)
    

    def placeInversion(self):
        for board, name, renderer in [('RF', 'inversion', self.renderRFBoard), ('slice', 'slice select inversion', self.renderSliceBoard)]:
            if name in self.boards[board]['objects']:
                sequence.moveWaveform(self.boards[board]['objects'][name], -self.TI)
                self.sequencePipeline.add(renderer)
        if 'inversion spoiler' in self.boards['slice']['objects']:
            spoilerTime = self.boards['RF']['objects']['inversion']['time'][-1] + self.boards['slice']['objects']['inversion spoiler']['dur_f']/2
            sequence.moveWaveform(self.boards['slice']['objects']['inversion spoiler'], spoilerTime)
        self.sequencePipeline.add(self.updateMinTR)
        self.sequencePipeline.add(self.updateBWbounds)
        self.sequencePipeline.add(self.renderFrequencyBoard) # due to TR bounds
        self.sequencePipeline.add(self.renderPhaseBoard) # due to TR bounds
        self.sequencePipeline.add(self.renderSliceBoard) # due to TR bounds
        self.sequencePipeline.add(self.renderRFBoard) # due to TR bounds
    

    def placeReadout(self):
        sequence.moveWaveform(self.boards['frequency']['objects']['readout'], self.TE)
        sequence.moveWaveform(self.boards['ADC']['objects']['sampling'], self.TE)
        if isGradientEcho(self.sequence):
            if self.boards['frequency']['objects']['read prephaser']['area_f'] > 0:
                sequence.rescaleGradient(self.boards['frequency']['objects']['read prephaser'], -1)
            prephaseTime = self.TE - sum([self.boards['frequency']['objects'][name]['dur_f'] for name in ['readout', 'read prephaser']])/2
        else:
            if self.boards['frequency']['objects']['read prephaser']['area_f'] < 0:
                sequence.rescaleGradient(self.boards['frequency']['objects']['read prephaser'], -1)
            prephaseTime = sum([self.boards[b]['objects'][name]['dur_f'] for (b, name) in [('RF', 'excitation'), ('frequency', 'read prephaser')]])/2
        sequence.moveWaveform(self.boards['frequency']['objects']['read prephaser'], prephaseTime)
        self.sequencePipeline.add(self.placePhaser)
        self.sequencePipeline.add(self.placeSpoiler)
        self.sequencePipeline.add(self.renderFrequencyBoard)
        self.sequencePipeline.add(self.updateMinTE)
    
    
    def placePhaser(self):
        phaserDur = self.boards['phase']['objects']['phase encode']['dur_f']
        if isGradientEcho(self.sequence):
            phaserTime = (self.boards['slice']['objects']['slice select excitation']['dur_f'] + phaserDur)/2
        else:
            phaserTime = self.TE - (self.boards['frequency']['objects']['readout']['flatDur_f'] + phaserDur)/2
        sequence.moveWaveform(self.boards['phase']['objects']['phase encode'], phaserTime)
        self.sequencePipeline.add(self.renderPhaseBoard)
    

    def placeSpoiler(self):
        spoilerTime = self.boards['frequency']['objects']['readout']['center_f'] + (self.boards['frequency']['objects']['readout']['flatDur_f'] + self.boards['slice']['objects']['spoiler']['dur_f']) / 2
        sequence.moveWaveform(self.boards['slice']['objects']['spoiler'], spoilerTime)
        self.sequencePipeline.add(self.renderSliceBoard)
        self.sequencePipeline.add(self.updateMinTR)


    def renderPolygons(self, board):
        self.boardPlots[board]['area'] = hv.Area(sequence.accumulateWaveforms(list(self.boards[board]['objects'].values()), board), self.timeDim, self.boards[board]['dim']).opts(color=self.boards[board]['color'])
        self.boards[board]['attributes'] = []
        if self.boards[board]['objects']:
            self.boards[board]['attributes'] += [attr for attr in list(self.boards[board]['objects'].values())[0].keys() if attr not in ['time', board] and '_f' not in attr]
            hover = HoverTool(tooltips=[(attr, '@{}'.format(attr)) for attr in self.boards[board]['attributes']], attachment='below')
            self.boardPlots[board]['polygons'] = hv.Polygons(list(self.boards[board]['objects'].values()), kdims=[self.timeDim, self.boards[board]['dim']], vdims=self.boards[board]['attributes']).opts(tools=[hover], cmap=[self.boards[board]['color']], hooks=[hideframe_hook, partial(bounds_hook, xbounds=(-19000, 19000))])
    
    
    def renderTRbounds(self, board):
        t0 = self.getSeqStart()
        self.boardPlots[board]['TRbounds'] = hv.VSpan(-20000, t0, kdims=[self.timeDim, self.boards[board]['dim']]).opts(color='gray', fill_alpha=.3)
        self.boardPlots[board]['TRbounds'] *= hv.VSpan(t0 + self.TR, 20000, kdims=[self.timeDim, self.boards[board]['dim']]).opts(color='gray', fill_alpha=.3)
    

    def renderFrequencyBoard(self):
        self.renderPolygons('frequency')
        self.boardPlots['frequency']['ADC'] = hv.Overlay([hv.VSpan(obj['time'][0], obj['time'][-1], kdims=[self.timeDim, self.boards['frequency']['dim']]) for obj in self.boards['ADC']['objects'].values()])
        self.renderTRbounds('frequency')
    

    def renderPhaseBoard(self):
        self.renderPolygons('phase')
        self.renderTRbounds('phase')
    

    def renderSliceBoard(self):
        self.renderPolygons('slice')
        self.renderTRbounds('slice')


    def renderRFBoard(self):
        self.renderPolygons('RF')
        self.renderTRbounds('RF')
    
    
    @param.depends('object', 'fieldStrength', 'sequence', 'FatSat', 'TR', 'TE', 'FA', 'TI', 'FOVF', 'FOVP', 'matrixF', 'matrixP', 'reconMatrixF', 'reconMatrixP', 'frequencyDirection', 'pixelBandWidth', 'NSA')
    def getKspace(self):
        self.runReconPipeline()
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
            coords={'kx': kAxes[1], 'ky': kAxes[0]}
        )
        ksp.kx.attrs['units'] = ksp.ky.attrs['units'] = '1/mm'
        return hv.Image(ksp, vdims=['magnitude'])
    

    @param.depends('object', 'fieldStrength', 'sequence', 'FatSat', 'TR', 'TE', 'FA', 'TI', 'FOVF', 'FOVP', 'matrixF', 'matrixP', 'reconMatrixF', 'reconMatrixP', 'frequencyDirection', 'pixelBandWidth', 'NSA')
    def getImage(self):
        self.runReconPipeline()
        iAxes = [(np.arange(self.reconMatrix[dim]) - (self.reconMatrix[dim]-1)/2) / self.reconMatrix[dim] * self.FOV[dim] for dim in range(2)]
        img = xr.DataArray(
            np.abs(self.imageArray), 
            dims=('y', 'x'),
            coords={'x': iAxes[1], 'y': iAxes[0][::-1]}
        )
        img.x.attrs['units'] = img.y.attrs['units'] = 'mm'
        return hv.Image(img, vdims=['magnitude'])
    

    @param.depends('sequence', 'FatSat', 'TR', 'TE', 'FA', 'TI', 'FOVF', 'FOVP', 'matrixF', 'matrixP', 'pixelBandWidth')
    def getSequencePlot(self):
        self.runSequencePipeline()
        return hv.Layout(list([hv.Overlay(list(boardPlot.values())).opts(border=0, xaxis='bottom' if n==len(self.boardPlots)-1 else None) for n, boardPlot in enumerate(self.boardPlots.values())])).cols(1).options(toolbar='below')


def hideShowButtonCallback(pane, event):
    if 'Show' in event.obj.name:
        pane.visible = True
        event.obj.name = event.obj.name.replace('Show', 'Hide')
    elif 'Hide' in event.obj.name:
        pane.visible = False
        event.obj.name = event.obj.name.replace('Hide', 'Show')


def getApp():
    explorer = MRIsimulator(name='')
    title = '# SpinSight MRI simulator'
    author = '*Written by [Johan Berglund](mailto:johan.berglund@akademiska.se), Ph.D.*'
    settingsParams = pn.panel(explorer.param, parameters=['object', 'fieldStrength'], name='Settings')
    contrastParams = pn.panel(explorer.param, parameters=['sequence', 'FatSat', 'TR', 'TE', 'FA', 'TI'], widgets={'TR': pn.widgets.DiscreteSlider, 'TE': pn.widgets.DiscreteSlider, 'TI': pn.widgets.DiscreteSlider}, name='Contrast')
    geometryParams = pn.panel(explorer.param, parameters=['FOVF', 'FOVP', 'matrixF', 'matrixP', 'reconMatrixF', 'reconMatrixP', 'frequencyDirection', 'pixelBandWidth', 'NSA'], name='Geometry')
    dmapKspace = pn.Row(hv.DynamicMap(explorer.getKspace), visible=False)
    dmapMRimage = hv.DynamicMap(explorer.getImage)
    dmapSequence = pn.Row(hv.DynamicMap(explorer.getSequencePlot), visible=False)
    sequenceButton = pn.widgets.Button(name='Show sequence')
    sequenceButton.on_click(partial(hideShowButtonCallback, dmapSequence))
    kSpaceButton = pn.widgets.Button(name='Show k-space')
    kSpaceButton.on_click(partial(hideShowButtonCallback, dmapKspace))
    dashboard = pn.Column(pn.Row(pn.Column(pn.pane.Markdown(title), pn.Row(pn.Column(settingsParams, pn.Row(sequenceButton, kSpaceButton), contrastParams), geometryParams)), dmapMRimage, dmapKspace), dmapSequence, pn.pane.Markdown(author))
    return dashboard


# TODO: add slice thickness
# TODO: phase oversampling
# TODO: abdomen phantom ribs, pancreas, hepatic arteries
# TODO: add params for matrix/pixelSize and BW like different vendors and handle their correlation
# TODO: add ACQ time and SNR
# TODO: add apodization
# TODO: parallel imaging (GRAPPA)
# TODO: B0 inhomogeneity
# TODO: Fast spin echo
# TODO: EPI