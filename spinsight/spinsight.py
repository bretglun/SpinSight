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
pn.config.theme = 'dark' # 'default' / 'dark'


GYRO = 42.577 # 1H gyromagnetic ratio [MHz/T]

TISSUES = {
    'gray':       {'PD': 1.0, 'FF': .00, 'T1': {1.5: 1100, 3.0: 1330}, 'T2': {1.5:   95, 3.0:   99}, 'hexcolor': '00ff00'},
    'white':      {'PD': 0.9, 'FF': .00, 'T1': {1.5:  560, 3.0:  830}, 'T2': {1.5:   72, 3.0:   69}, 'hexcolor': 'd40000'},
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
    'abdomen': {'FOV': (320, 400), 'matrix': (513, 641), 'referenceTissue': 'spleen'}, # odd matrix to ensure kspace center is sampled (not required)
    'brain': {'FOV': (188, 156), 'matrix': (601, 601), 'referenceTissue': 'gray'} # odd matrix to ensure kspace center is sampled (not required)
}

DIRECTIONS = {'anterior-posterior': 0, 'left-right': 1}


def pixelBW2shift(pixelBW, B0=1.5):
    ''' Get fat/water chemical shift [pixels] from pixel bandwidth [Hz/pixel] and B0 [T]'''
    return np.abs(FATRESONANCES['Fat2']['shift'] * GYRO * B0 / pixelBW)


def shift2pixelBW(shift, B0=1.5):
    ''' Get pixel bandwidth [Hz/pixel] from fat/water chemical shift [pixels] and B0 [T]'''
    return np.abs(FATRESONANCES['Fat2']['shift'] * GYRO * B0 / shift)


def pixelBW2FOVBW(pixelBW, matrixF):
    ''' Get FOV bandwidth [±kHz] from pixel bandwidth [Hz/pixel] and read direction matrix'''
    return pixelBW * matrixF / 2e3


def FOVBW2pixelBW(FOVBW, matrixF):
    ''' Get pixel bandwidth [Hz/pixel] from FOV bandwidth [±kHz] and read direction matrix'''
    return FOVBW / matrixF * 2e3


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
    for dim, n in enumerate(kspace.shape):
        shape = tuple(reconMatrix[dim] - n if d==0 else 1 for d in range(kspace.ndim))
        kspace = np.insert(kspace, n-n//2, np.zeros(shape), axis=dim)
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


def getBounds(minval, maxval, curval):
    if curval < minval:
        print('Warning: trying to set bounds above current value ({} > {})'.format(minval, curval))
        minval = curval
    if curval > maxval:
        print('Warning: trying to set bounds below current value ({} < {})'.format(maxval, curval))
        maxval = curval
    return (minval, maxval)


def bounds_hook(plot, elem, xbounds=None):
    x_range = plot.handles['plot'].x_range
    if xbounds is not None:
        x_range.bounds = xbounds
    else:
        x_range.bounds = x_range.start, x_range.end 


def hideframe_hook(plot, elem):
    plot.handles['plot'].outline_line_color = None


def flatten_dicts(list_of_dicts_and_lists):
    res = []
    for v in list_of_dicts_and_lists:
        res += flatten_dicts(v) if isinstance(v, list) else [v]
    return res


TRvalues = [float('{:.2g}'.format(tr)) for tr in 10.**np.linspace(0, 4, 500)]
TEvalues = [float('{:.2g}'.format(te)) for te in 10.**np.linspace(0, 3, 500)]
TIvalues = [float('{:.2g}'.format(ti)) for ti in 10.**np.linspace(0, 4, 500)]
matrixValues = list(range(16, 600+1))
EPIfactorValues = list(range(1, 64+1))


class MRIsimulator(param.Parameterized):
    object = param.ObjectSelector(default='brain', objects=PHANTOMS.keys(), label='Phantom object')
    fieldStrength = param.ObjectSelector(default=1.5, objects=[1.5, 3.0], label='B0 field strength [T]')
    parameterStyle = param.ObjectSelector(default='Siemens', objects=['Siemens', 'Philips', 'GE'], label='Parameter Style')
    
    FatSat = param.Boolean(default=False, label='Fat saturation')
    TR = param.Selector(default=10000, objects=TRvalues, label='TR [msec]')
    TE = param.Selector(default=10, objects=TEvalues, label='TE [msec]')
    FA = param.Number(default=90.0, bounds=(1, 90.0), precedence=-1, label='Flip angle [°]')
    TI = param.Selector(default=40, objects=TIvalues, precedence=-1, label='TI [msec]')
    
    frequencyDirection = param.ObjectSelector(default=list(DIRECTIONS.keys())[0], objects=DIRECTIONS.keys(), precedence=1, label='Frequency encoding direction')
    FOVP = param.Number(default=240, bounds=(100, 600), precedence=2, label='FOV x [mm]')
    FOVF = param.Number(default=240, bounds=(100, 600), precedence=2, label='FOV y [mm]')
    phaseOversampling = param.Number(default=0, bounds=(0, 100), step=1., precedence=3, label='Phase oversampling [%]')
    voxelP = param.Selector(default=1.333, precedence=-4, label='Voxel size x [mm]')
    voxelF = param.Selector(default=1.333, precedence=-4, label='Voxel size y [mm]')
    matrixP = param.Selector(default=180, objects=matrixValues, precedence=4, label='Acquisition matrix x')
    matrixF = param.Selector(default=180, objects=matrixValues, precedence=4, label='Acquisition matrix y')
    reconVoxelP = param.Selector(default=0.666, precedence=-5, label='Reconstructed voxel size x [mm]')
    reconVoxelF = param.Selector(default=0.666, precedence=-5, label='Reconstructed voxel size y [mm]')
    reconMatrixP = param.Integer(default=360, bounds=(matrixP.default, 1024), precedence=5, label='Reconstruction matrix x')
    reconMatrixF = param.Integer(default=360, bounds=(matrixF.default, 1024), precedence=5, label='Reconstruction matrix y')
    sliceThickness = param.Number(default=3, bounds=(0.5, 10), precedence=6, label='Slice thickness [mm]')
    
    sequence = param.ObjectSelector(default=SEQUENCES[0], objects=SEQUENCES, precedence=1, label='Pulse sequence')
    pixelBandWidth = param.Number(default=500, bounds=(125, 2000), precedence=2, label='Pixel bandwidth [Hz]')
    FOVbandwidth = param.Number(default=pixelBW2FOVBW(500, 180), bounds=(pixelBW2FOVBW(125, 180), pixelBW2FOVBW(2000, 180)), precedence=-2, label='FOV bandwidth [±kHz]')
    FWshift = param.Number(default=pixelBW2shift(500), bounds=(pixelBW2shift(2000), pixelBW2shift(125)), precedence=-2, label='Fat/water shift [pixels]')
    NSA = param.Integer(default=1, bounds=(1, 16), precedence=3, label='NSA')
    kspaceOrder = param.ObjectSelector(default='center echo', objects=['early echo', 'center echo', 'late echo'], precedence=4, label='k-space order')
    partialFourier = param.Number(default=1, bounds=(.6, 1), step=0.01, precedence=5, label='Partial Fourier factor')
    turboFactor = param.Integer(default=1, bounds=(1, 64), precedence=6, label='Turbo factor')
    EPIfactor = param.Selector(default=1, objects=EPIfactorValues, precedence=7, label='EPI factor')
    
    showFOV = param.Boolean(default=False, label='Show FOV')
    noiseGain = param.Number(default=3.)
    SNR = param.Number(label='SNR')
    referenceSNR = param.Number(default=1, label='Reference SNR')
    relativeSNR = param.Number(label='Relative SNR [%]')
    scantime = param.Number(label='Scan time [sec]')

    def __init__(self, **params):
        super().__init__(**params)

        self.render = True # semaphore to avoid unneccesary rendering

        self.fullReconPipeline = [
            self.loadPhantom, 
            self.sampleKspace, 
            self.updateSamplingTime, 
            self.modulateKspace, 
            self.simulateNoise, 
            self.updatePDandT1w, 
            self.compileKspace, 
            self.partialFourierRecon,
            self.zerofill, 
            self.reconstruct,
            self.setReferenceSNR
        ]
        
        self.reconPipeline = set(self.fullReconPipeline)

        self.timeDim = hv.Dimension('time', label='time', unit='ms')

        self.boards = { 'frequency': {'dim': hv.Dimension('frequency', label='G read', unit='mT/m', range=(-30, 30)), 'color': 'cadetblue'}, 
                        'phase': {'dim': hv.Dimension('phase', label='G phase', unit='mT/m', range=(-30, 30)), 'color': 'cadetblue'}, 
                        'slice': {'dim': hv.Dimension('slice', label='G slice', unit='mT/m', range=(-30, 30)), 'color': 'cadetblue'}, 
                        'RF': {'dim': hv.Dimension('RF', label='RF', unit='μT', range=(-5, 25)), 'color': 'red'},
                        'ADC': {'dim': hv.Dimension('ADC', label='ADC', unit=''), 'color': 'orange'} }

        self._watch_reconMatrixF()
        self._watch_reconMatrixP()

        self.boardPlots = {board: {'hline': hv.HLine(0.0, kdims=[self.timeDim, self.boards[board]['dim']]).opts(tools=['xwheel_zoom', 'xpan', 'reset'], default_tools=[], active_tools=['xwheel_zoom', 'xpan'])} for board in self.boards if board != 'ADC'}

        hv.opts.defaults(hv.opts.Image(width=500, height=500, invert_yaxis=False, toolbar='below', cmap='gray', aspect='equal'))
        hv.opts.defaults(hv.opts.HLine(line_width=1.5, line_color='gray'))
        hv.opts.defaults(hv.opts.VSpan(color='orange', fill_alpha=.1, hover_fill_alpha=.8, default_tools=[]))
        hv.opts.defaults(hv.opts.Box(line_width=3))
        hv.opts.defaults(hv.opts.Area(fill_alpha=.5, line_width=1.5, line_color='gray', default_tools=[]))
        hv.opts.defaults(hv.opts.Polygons(line_width=1.5, fill_alpha=0, line_alpha=0, line_color='gray', selection_line_color='black', hover_fill_alpha=.8, hover_line_alpha=1, selection_fill_alpha=.8, selection_line_alpha=1, nonselection_line_alpha=0, default_tools=[]))

        self.maxAmp = 25. # mT/m
        self.maxSlew = 80. # T/m/s
        self.inversionThkFactor = 1.1 # make inversion slice 10% thicker

        for board in self.boards:
            self.boards[board]['objects'] = {}

        self.fullSequencePipeline = [
            self.setupExcitation, 
            self.setupRefocusing,
            self.setupInversion,
            self.setupFatSat,
            self.setupSliceSelection,
            self.setupReadouts,
            self.setupPhaser,
            self.setupSpoiler,
            self.placeRefocusing,
            self.placeInversion,
            self.placeFatSat,
            self.placeReadouts,
            self.placePhaser,
            self.placeSpoiler,
            self.updateMinTE,
            self.updateMinTR,
            self.updateMaxTE,
            self.updateMaxTI,
            self.updateBWbounds,
            self.updateMatrixFbounds,
            self.updateMatrixPbounds,
            self.updateFOVFbounds, 
            self.updateFOVPbounds,
            self.updateSliceThicknessBounds,
            self.renderFrequencyBoard, 
            self.renderPhaseBoard, 
            self.renderSliceBoard, 
            self.renderRFBoard
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
        minFOV = PHANTOMS[self.object]['FOV']
        if self.frequencyDirection=='left-right':
            minFOV = minFOV.reverse()
        self.FOVF = max(self.FOVF, minFOV[0])
        self.FOVP = max(self.FOVP, minFOV[1])
        self.reconPipeline.add(self.setReferenceSNR)
        self.runReconPipeline() # rerun recon pipeline to set referenceSNR after FOV update
    

    @param.depends('parameterStyle', watch=True)
    def _watch_parameterStyle(self):
        for param in [self.param.voxelF, self.param.voxelP, self.param.matrixF, self.param.matrixP, self.param.reconVoxelF, self.param.reconVoxelP, self.param.reconMatrixF, self.param.reconMatrixP, self.param.pixelBandWidth, self.param.FOVbandwidth, self.param.FWshift]:
            param.precedence = -1
        if self.parameterStyle == 'Philips':
            self.param.voxelF.precedence = 4
            self.param.voxelP.precedence = 4
            self.param.reconVoxelF.precedence = 5
            self.param.reconVoxelP.precedence = 5
            self.param.FWshift.precedence = 2
        else:
            self.param.matrixF.precedence = 4
            self.param.matrixP.precedence = 4
            self.param.reconMatrixF.precedence = 5
            self.param.reconMatrixP.precedence = 5
            if self.parameterStyle == 'Siemens':
                self.param.pixelBandWidth.precedence = 2
            elif self.parameterStyle == 'GE':
                self.param.FOVbandwidth.precedence = 2
                self.updateMatrixFbounds()


    @param.depends('FOVF', watch=True)
    def _watch_FOVF(self):
        if self.parameterStyle=='Philips': # Philips style, update matrix
            self.matrixF = int(np.round(self.FOVF / self.voxelF))
            self.reconMatrixF = int(np.round(self.FOVF / self.reconVoxelF))
        self.updateVoxelFobjects()
        self.updateReconVoxelFobjects()
        self.voxelF = min(self.param.voxelF.objects, key=lambda x: abs(x-self.FOVF/self.matrixF))
        self.reconVoxelF = min(self.param.reconVoxelF.objects, key=lambda x: abs(x-self.FOVF/self.reconMatrixF))
        for f in [self.sampleKspace, self.updateSamplingTime, self.modulateKspace, self.simulateNoise, self.compileKspace, self.partialFourierRecon, self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)
        for f in [self.setupReadouts, self.updateBWbounds, self.updateMatrixFbounds]:
            self.sequencePipeline.add(f)
    

    @param.depends('FOVP', watch=True)
    def _watch_FOVP(self):
        if self.parameterStyle=='Philips': # Philips style, update matrix
            self.matrixP = int(np.round(self.FOVP / self.voxelP))
            self.reconMatrixP = int(np.round(self.FOVP / self.reconVoxelP))
        self.updateVoxelPobjects()
        self.updateReconVoxelPobjects()
        self.voxelP = min(self.param.voxelP.objects, key=lambda x: abs(x-self.FOVP/self.matrixP))
        self.reconVoxelP = min(self.param.reconVoxelP.objects, key=lambda x: abs(x-self.FOVP/self.reconMatrixP))
        for f in [self.sampleKspace, self.updateSamplingTime, self.modulateKspace, self.simulateNoise, self.compileKspace, self.partialFourierRecon, self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)
        for f in [self.setupPhaser, self.updateMatrixPbounds]:
            self.sequencePipeline.add(f)


    @param.depends('phaseOversampling', watch=True)
    def _watch_phaseOversampling(self):
        self._watch_FOVP()


    @param.depends('matrixF', watch=True)
    def _watch_matrixF(self):
        self.param.FOVbandwidth.bounds = getBounds(pixelBW2FOVBW(self.param.pixelBandWidth.bounds[0], self.matrixF), pixelBW2FOVBW(self.param.pixelBandWidth.bounds[1], self.matrixF), self.FOVbandwidth)
        if self.parameterStyle == 'GE':
            self.pixelBandWidth = FOVBW2pixelBW(self.FOVbandwidth, self.matrixF)
        else:
            self.FOVbandwidth = pixelBW2FOVBW(self.pixelBandWidth, self.matrixF)
        self.voxelF = min(self.param.voxelF.objects, key=lambda x: abs(x-self.FOVF/self.matrixF))
        for f in [self.sampleKspace, self.updateSamplingTime, self.modulateKspace, self.simulateNoise, self.compileKspace, self.partialFourierRecon, self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)
        for f in [self.setupReadouts, self.updateBWbounds, self.updateMatrixFbounds, self.updateFOVFbounds]:
            self.sequencePipeline.add(f)
        self.param.reconMatrixF.bounds = (self.matrixF, self.param.reconMatrixF.bounds[1])
        self.updateReconVoxelFobjects()
        self.reconMatrixF = min(max(int(np.round(self.matrixF * self.recAcqRatioF)), self.matrixF), self.param.reconMatrixF.bounds[1])
    
    
    @param.depends('matrixP', watch=True)
    def _watch_matrixP(self):
        self.voxelP = min(self.param.voxelP.objects, key=lambda x: abs(x-self.FOVP/self.matrixP))
        for f in [self.sampleKspace, self.updateSamplingTime, self.modulateKspace, self.simulateNoise, self.compileKspace, self.partialFourierRecon, self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)
        for f in [self.setupPhaser, self.updateFOVPbounds]:
            self.sequencePipeline.add(f)
        self.param.reconMatrixP.bounds = (self.matrixP, self.param.reconMatrixP.bounds[1])
        self.updateReconVoxelPobjects()
        self.reconMatrixP = min(max(int(np.round(self.matrixP * self.recAcqRatioP)), self.matrixP), self.param.reconMatrixP.bounds[1])


    @param.depends('voxelF', watch=True)
    def _watch_voxelF(self):
        self.matrixF = int(np.round(self.FOVF/self.voxelF))


    @param.depends('voxelP', watch=True)
    def _watch_voxelP(self):
        self.matrixP = int(np.round(self.FOVP/self.voxelP))


    @param.depends('reconVoxelF', watch=True)
    def _watch_reconVoxelF(self):
        self.reconMatrixF = int(np.round(self.FOVF/self.reconVoxelF))


    @param.depends('reconVoxelP', watch=True)
    def _watch_reconVoxelP(self):
        self.reconMatrixP = int(np.round(self.FOVP/self.reconVoxelP))


    @param.depends('sliceThickness', watch=True)
    def _watch_sliceThickness(self):
        for f in [self.sampleKspace, self.updateSamplingTime, self.modulateKspace, self.simulateNoise, self.compileKspace, self.partialFourierRecon, self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)
        for f in [self.setupSliceSelection, self.placeFatSat]:
            self.sequencePipeline.add(f)
    
    
    @param.depends('frequencyDirection', watch=True)
    def _watch_frequencyDirection(self):
        for f in [self.sampleKspace, self.updateSamplingTime, self.modulateKspace, self.simulateNoise, self.compileKspace, self.partialFourierRecon, self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)
        for p in [self.param.FOVF, self.param.FOVP, self.param.matrixF, self.param.matrixP, self.param.reconMatrixF, self.param.reconMatrixP]:
            if ' x' in p.label:
                p.label = p.label.replace(' x', ' y')
            elif ' y' in p.label:
                p.label = p.label.replace(' y', ' x')


    @param.depends('fieldStrength', watch=True)
    def _watch_fieldStrength(self):
        self.updateBWbounds()
        self.FWshift = pixelBW2shift(self.pixelBandWidth, self.fieldStrength)
        for f in [self.updateSamplingTime, self.modulateKspace, self.simulateNoise, self.updatePDandT1w, self.compileKspace, self.partialFourierRecon, self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)
        self._watch_FatSat() # since fatsat pulse duration depends on fieldStrength
    

    @param.depends('pixelBandWidth', watch=True)
    def _watch_pixelBandWidth(self):
        self.FWshift = pixelBW2shift(self.pixelBandWidth, self.fieldStrength)
        self.FOVbandwidth = pixelBW2FOVBW(self.pixelBandWidth, self.matrixF)
        for f in [self.updateSamplingTime, self.modulateKspace, self.simulateNoise, self.compileKspace, self.partialFourierRecon, self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)
        for f in [self.setupReadouts, self.updateMatrixFbounds, self.updateFOVFbounds, self.updateMatrixPbounds, self.updateFOVPbounds]:
            self.sequencePipeline.add(f)
    
    @param.depends('FWshift', watch=True)
    def _watch_FWshift(self):
        self.pixelBandWidth = shift2pixelBW(self.FWshift, self.fieldStrength)
    

    @param.depends('FOVbandwidth', watch=True)
    def _watch_FOVbandwidth(self):
        self.pixelBandWidth = FOVBW2pixelBW(self.FOVbandwidth, self.matrixF)


    @param.depends('NSA', watch=True)
    def _watch_NSA(self):
        for f in [self.updateSamplingTime, self.modulateKspace, self.simulateNoise, self.compileKspace, self.partialFourierRecon, self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)


    @param.depends('kspaceOrder', watch=True)
    def _watch_kspaceOrder(self):
        for f in [self.modulateKspace, self.simulateNoise, self.compileKspace, self.partialFourierRecon, self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)
        for f in [self.placeRefocusing, self.placeReadouts, self.placePhaser, self.updateMinTE, self.updateMinTR, self.updateMaxTE, self.updateMaxTI, self.updateBWbounds, self.updateMatrixFbounds, self.updateMatrixPbounds, self.updateFOVFbounds,  self.updateFOVPbounds, self.updateSliceThicknessBounds]:
            self.sequencePipeline.add(f)


    @param.depends('partialFourier', watch=True)
    def _watch_partialFourier(self):
        for f in [self.sampleKspace, self.updateSamplingTime, self.modulateKspace, self.simulateNoise, self.compileKspace, self.partialFourierRecon, self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)
        for f in [self.setupRefocusing, self.setupReadouts, self.setupPhaser, self.updateMinTE, self.updateMinTR, self.updateMaxTE, self.updateMaxTI, self.updateBWbounds, self.updateMatrixFbounds, self.updateMatrixPbounds, self.updateFOVFbounds,  self.updateFOVPbounds, self.updateSliceThicknessBounds]:
            self.sequencePipeline.add(f)


    @param.depends('turboFactor', watch=True)
    def _watch_turboFactor(self):
        for f in [self.sampleKspace, self.updateSamplingTime, self.modulateKspace, self.simulateNoise, self.compileKspace, self.partialFourierRecon, self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)
        for f in [self.setupRefocusing, self.setupReadouts, self.setupPhaser, self.updateMinTE, self.updateMinTR, self.updateMaxTE, self.updateMaxTI, self.updateBWbounds, self.updateMatrixFbounds, self.updateMatrixPbounds, self.updateFOVFbounds,  self.updateFOVPbounds, self.updateSliceThicknessBounds]:
            self.sequencePipeline.add(f)
        self.updateEPIfactorObjects()


    @param.depends('EPIfactor', watch=True)
    def _watch_EPIfactor(self):
        for f in [self.sampleKspace, self.updateSamplingTime, self.modulateKspace, self.simulateNoise, self.compileKspace, self.partialFourierRecon, self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)
        for f in [self.setupReadouts, self.setupPhaser, self.updateMinTE, self.updateMinTR, self.updateMaxTE, self.updateMaxTI, self.updateBWbounds, self.updateMatrixFbounds, self.updateMatrixPbounds, self.updateFOVFbounds,  self.updateFOVPbounds, self.updateSliceThicknessBounds]:
            self.sequencePipeline.add(f)
        self.updateTurboFactorBounds()


    @param.depends('sequence', watch=True)
    def _watch_sequence(self):
        for f in [self.modulateKspace, self.updatePDandT1w, self.compileKspace, self.partialFourierRecon, self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)
        for f in [self.setupExcitation, self.setupRefocusing, self.setupInversion, self.placeReadouts, self.placePhaser]:
            self.sequencePipeline.add(f)
        self.param.FA.precedence = 1 if self.sequence=='Spoiled Gradient Echo' else -1
        self.param.TI.precedence = 1 if self.sequence=='Inversion Recovery' else -1
        tr = self.TR
        self.render = False
        self.TR = self.param.TR.objects[-1] # max TR
        self.runSequencePipeline()
        self.TE = min(self.param.TE.objects, key=lambda x: abs(x-self.TE)) # TE within bounds
        self.runSequencePipeline()
        if self.sequence=='Inversion Recovery':
            self.TI = min(self.param.TI.objects, key=lambda x: abs(x-self.TI)) # TI within bounds
            self.runSequencePipeline()
        self.render = True
        self.TR = min(self.param.TR.objects, key=lambda x: abs(x-tr)) # Set back TR within bounds
    

    @param.depends('TE', watch=True)
    def _watch_TE(self):
        for f in [self.modulateKspace, self.updatePDandT1w, self.compileKspace, self.partialFourierRecon, self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)
        for f in [self.placeRefocusing, self.placeReadouts, self.placePhaser, self.updateMatrixFbounds, self.updateFOVFbounds, self.updateMatrixPbounds, self.updateFOVPbounds]:
            self.sequencePipeline.add(f)
    

    @param.depends('TR', watch=True)
    def _watch_TR(self):
        for f in [self.updatePDandT1w, self.compileKspace, self.partialFourierRecon, self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)
        for f in [self.updateMaxTE, self.updateMaxTI, self.updateBWbounds, self.updateSliceThicknessBounds, self.renderFrequencyBoard, self.renderPhaseBoard, self.renderSliceBoard, self.renderRFBoard]:
            self.sequencePipeline.add(f)
    

    @param.depends('TI', watch=True)
    def _watch_TI(self):
        for f in [self.updatePDandT1w, self.compileKspace, self.partialFourierRecon, self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)
        self.sequencePipeline.add(self.placeInversion)


    @param.depends('FA', watch=True)
    def _watch_FA(self):
        for f in [self.updatePDandT1w, self.compileKspace, self.partialFourierRecon, self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)
        self.sequencePipeline.add(self.setupExcitation)
    
    
    @param.depends('FatSat', watch=True)
    def _watch_FatSat(self):
        for f in [self.compileKspace, self.partialFourierRecon, self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)
        for f in [self.setupFatSat, self.updateMaxTE, self.updateBWbounds]:
            self.sequencePipeline.add(f)
        tr = self.TR
        self.render = False
        self.TR = self.param.TR.objects[-1] # max TR
        self.runSequencePipeline()
        self.render = True
        self.TR = min(self.param.TR.objects, key=lambda x: abs(x-tr)) # Set back TR within bounds


    @param.depends('reconMatrixF', watch=True)
    def _watch_reconMatrixF(self):
        self.recAcqRatioF = self.reconMatrixF / self.matrixF
        self.reconVoxelF = min(self.param.reconVoxelF.objects, key=lambda x: abs(x-self.FOVF/self.reconMatrixF))
        for f in [self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)


    @param.depends('reconMatrixP', watch=True)
    def _watch_reconMatrixP(self):
        self.recAcqRatioP = self.reconMatrixP / self.matrixP
        self.reconVoxelP = min(self.param.reconVoxelP.objects, key=lambda x: abs(x-self.FOVP/self.reconMatrixP))
        for f in [self.zerofill, self.reconstruct]:
            self.reconPipeline.add(f)


    def getSeqStart(self):
        if self.sequence == 'Inversion Recovery': 
            return self.boards['slice']['objects']['slice select inversion']['time'][0]
        elif self.FatSat:
            return self.boards['RF']['objects']['fatsat']['time'][0]
        else:
            return self.boards['slice']['objects']['slice select excitation']['time'][0]
    
    
    def updateMinTE(self):
        if not isGradientEcho(self.sequence):
            leftSide = max(
                self.boards['frequency']['objects']['read prephaser']['dur_f'], 
                self.boards['slice']['objects']['slice select excitation']['riseTime_f'] + self.boards['slice']['objects']['slice select rephaser']['dur_f'] + (self.boards['slice']['objects']['slice select refocusing'][0]['riseTime_f']))
            leftSide += (self.boards['RF']['objects']['excitation']['dur_f'] + self.boards['RF']['objects']['refocusing'][0]['dur_f']) / 2
            rightSide = max(
                self.boards['frequency']['objects']['readouts'][0][0]['riseTime_f'],
                self.boards['phase']['objects']['phase encode']['dur_f'],
                self.boards['slice']['objects']['slice select refocusing'][0]['riseTime_f'])
            rightSide += (self.boards['RF']['objects']['refocusing'][0]['dur_f'] + self.boards['ADC']['objects']['samplings'][0][0]['dur_f']) / 2
            self.minTE = max(leftSide, rightSide) * 2
        else:
            self.minTE = max(
                self.boards['frequency']['objects']['read prephaser']['dur_f'] + self.boards['frequency']['objects']['readouts'][0][0]['riseTime_f'],
                self.boards['phase']['objects']['phase encode']['dur_f'],
                self.boards['slice']['objects']['slice select excitation']['riseTime_f'] + self.boards['slice']['objects']['slice select rephaser']['dur_f']
            )
            self.minTE += (self.boards['RF']['objects']['excitation']['dur_f'] + self.boards['ADC']['objects']['samplings'][0][0]['dur_f']) / 2
        self.sequencePipeline.add(self.updateMaxTE)
    
    
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
        self.param.TI.objects, _ = updateBounds(self.TI, TIvalues, minval=40, maxval=maxTI)
    
    
    def getMaxPrephaserArea(self, readAmp):
        if isGradientEcho(self.sequence):
            maxPrephaserDur =  self.TE - self.boards['ADC']['objects']['samplings'][0][0]['dur_f']/2 - self.boards['RF']['objects']['excitation']['dur_f']/2 - readAmp/self.maxSlew
        else:
            maxPrephaserDur =  self.TE/2 - self.boards['RF']['objects']['refocusing'][0]['dur_f']/2 - self.boards['RF']['objects']['excitation']['dur_f']/2
        maxPrephaserFlatDur = maxPrephaserDur - (2 * self.maxAmp/self.maxSlew)
        if maxPrephaserFlatDur < 0: # triangle
            maxPrephaserArea = maxPrephaserDur**2 * self.maxSlew / 4
        else: # trapezoid
            slewArea = self.maxAmp**2 / self.maxSlew
            flatArea = self.maxAmp * maxPrephaserFlatDur
            maxPrephaserArea = slewArea + flatArea
        return maxPrephaserArea
    

    def getMaxReadoutArea(self):
        # See readouts.tex for formulae
        d = 1e3 / self.pixelBandWidth # readout duration
        s = self.maxSlew
        if isGradientEcho(self.sequence):
            t = self.TE - self.boards['RF']['objects']['excitation']['dur_f']/2
            h = s * (t - np.sqrt(t**2/2 + d**2/8)) # negative sqrt seems to be the reasonable solution
            h = min(h, self.maxAmp)
            A = d * (np.sqrt((d*s+2*h)**2 - 8*h*(h-s*(t-d/2))) - d*s - 2*h) / 2
            maxReadoutArea1 = A
        else: # spin echo
            tr = self.TE/2 - self.boards['RF']['objects']['refocusing'][0]['dur_f']/2
            tp = self.TE/2 - self.boards['RF']['objects']['refocusing'][0]['dur_f']/2 - self.boards['RF']['objects']['excitation']['dur_f']/2
            Ar = d*s* tr - d**2*s/2
            h = s * tp / 2
            h = min(h, self.maxAmp)
            Ap = d * (np.sqrt((d*s)**2 - 8*h*(h-s*tp)) - d*s) / 2
            maxReadoutArea1 = min(Ar, Ap)
        maxReadoutArea2 = self.maxAmp * 1e3 / self.pixelBandWidth # max wrt maxAmp
        return min(maxReadoutArea1, maxReadoutArea2)
    

    def getMaxPhaserArea(self):
        readStart = self.TE - 1e3 / self.pixelBandWidth / 2
        if isGradientEcho(self.sequence):
            maxPhaserDuration = readStart - self.boards['RF']['objects']['excitation']['dur_f']/2
        else:
            maxPhaserDuration =  readStart - (self.TE/2 + self.boards['RF']['objects']['refocusing'][0]['dur_f']/2)
        maxRiseTime = self.maxAmp / self.maxSlew
        if maxPhaserDuration > 2 * maxRiseTime: # trapezoid maxPhaser
            maxPhaserArea = (maxPhaserDuration - maxRiseTime) * self.maxAmp
        else: # triangular maxPhaser
            maxPhaserArea = (maxPhaserDuration/2)**2 * self.maxSlew
        return maxPhaserArea
    

    def updateBWbounds(self):
        # See readouts.tex for formulae relating to the readout board
        s = self.maxSlew
        A = 1e3 * self.matrixF / (self.FOVF * GYRO) # readout area
        minReadDurations = [.5] # msec (corresponds to a pixel BW of 2000 Hz)
        # min limit imposed by maximum gradient amplitude:
        minReadDurations.append(A / self.maxAmp)
        maxReadDurations = [8.] # msec (corresponds to a pixel BW of 125 Hz)
        # max limit imposed by TR:
        maxReadDurations.append((self.TR - (self.TE - self.getSeqStart()) - self.boards['slice']['objects']['spoiler']['dur_f']) * 2)
        if isGradientEcho(self.sequence):
            freeSpaceLeft = self.TE - self.boards['RF']['objects']['excitation']['dur_f']/2
            freeSpaceLeft -= max(
                self.boards['frequency']['objects']['read prephaser']['dur_f'] + self.boards['frequency']['objects']['readouts'][0][0]['riseTime_f'], # TODO: consider maximum dur+risetime, not only current (difficult!)
                self.boards['phase']['objects']['phase encode']['dur_f'],
                self.boards['slice']['objects']['slice select excitation']['riseTime_f'] + self.boards['slice']['objects']['slice select rephaser']['dur_f'])
            maxReadDurations.append(freeSpaceLeft*2)
        else: # spin echo
            # min limit imposed by prephaser duration tp:
            tp = self.TE/2 - self.boards['RF']['objects']['refocusing'][0]['dur_f']/2 - self.boards['RF']['objects']['excitation']['dur_f']/2
            h = s * tp / 2
            h = min(h, self.maxAmp)
            minReadDurations.append(np.sqrt(A**2/(2*h*s*tp - s*A - 2*h**2)))
            # maxlimit imposed by readout rise time:
            tr = (self.TE - self.boards['RF']['objects']['refocusing'][0]['dur_f'])/2
            maxReadDurations.append(tr + np.sqrt(tr**2 - 2*A/s))
            # max limit imposed by phaser:
            maxReadDurations.append((tr - self.boards['phase']['objects']['phase encode']['dur_f']) * 2)
            # max limit imposed by slice select refocusing down ramp time:
            maxReadDurations.append((tr - self.boards['slice']['objects']['slice select refocusing'][0]['riseTime_f']) * 2)
        minpBW = 1e3 / min(maxReadDurations)
        maxpBW = 1e3 / max(minReadDurations)
        self.param.pixelBandWidth.bounds = getBounds(minpBW, maxpBW, self.pixelBandWidth)
        self.param.FWshift.bounds = getBounds(pixelBW2shift(maxpBW, self.fieldStrength), pixelBW2shift(minpBW, self.fieldStrength), self.FWshift)
        self.param.FOVbandwidth.bounds = getBounds(pixelBW2FOVBW(minpBW, self.matrixF), pixelBW2FOVBW(maxpBW, self.matrixF), self.FOVbandwidth)
    

    def updateMatrixFbounds(self):
        minMatrixF, maxMatrixF = 16, 600
        maxMatrixF = min(maxMatrixF, int(self.getMaxReadoutArea() * 1e-3 * self.FOVF * GYRO))
        if self.parameterStyle == 'GE':
            minMatrixF = max(minMatrixF, int(np.ceil(self.FOVbandwidth * 2e3 / self.param.pixelBandWidth.bounds[1])))
            maxMatrixF = min(maxMatrixF, int(np.floor(self.FOVbandwidth * 2e3 / self.param.pixelBandWidth.bounds[0])))
        self.param.matrixF.objects, _ = updateBounds(self.matrixF, matrixValues, minval=minMatrixF, maxval=maxMatrixF)
        self.updateVoxelFobjects()
        self.updateReconVoxelFobjects()
    

    def updateMatrixPbounds(self):
        maxMatrixP = int(self.getMaxPhaserArea() * 2e-3 * self.FOVP * GYRO)
        self.param.matrixP.objects, _ = updateBounds(self.matrixP, matrixValues, maxval=maxMatrixP)
        self.updateVoxelPobjects()
        self.updateReconVoxelPobjects()


    def updateFOVFbounds(self):
        minFOVF = 1e3 * self.matrixF / (self.getMaxReadoutArea() * GYRO)
        self.param.FOVF.bounds = getBounds(max(minFOVF, 100), 600, self.FOVF)


    def updateFOVPbounds(self):
        minFOVP = 1e3 * self.matrixP / (self.getMaxPhaserArea() * GYRO * 2)
        self.param.FOVP.bounds = getBounds(max(minFOVP, 100), 600, self.FOVP)


    def updateVoxelFobjects(self):
        self.param.voxelF.objects = [float('{:.4g}'.format(self.FOVF/matrix)) for matrix in self.param.matrixF.objects[::-1]]


    def updateVoxelPobjects(self):
        self.param.voxelP.objects = [float('{:.4g}'.format(self.FOVP/matrix)) for matrix in self.param.matrixP.objects[::-1]]


    def updateReconVoxelFobjects(self):
        self.param.reconVoxelF.objects = [float('{:.4g}'.format(self.FOVF/matrix)) for matrix in range(self.param.reconMatrixF.bounds[1], self.param.reconMatrixF.bounds[0]-1, -1)]


    def updateReconVoxelPobjects(self):
        self.param.reconVoxelP.objects = [float('{:.4g}'.format(self.FOVP/matrix)) for matrix in range(self.param.reconMatrixP.bounds[1], self.param.reconMatrixP.bounds[0]-1, -1)]


    def updateSliceThicknessBounds(self):
        minThks = [.5]
        minThks.append(self.boards['RF']['objects']['excitation']['FWHM_f'] / (self.maxAmp * GYRO))
        if not isGradientEcho(self.sequence):
            minThks.append(self.boards['RF']['objects']['refocusing'][0]['FWHM_f'] / (self.maxAmp * GYRO))
        if self.sequence=='Inversion Recovery':
            minThks.append(self.boards['RF']['objects']['inversion']['FWHM_f'] / (self.maxAmp * GYRO) * self.inversionThkFactor)
        
        # Constraint due to TR: 
        if self.sequence=='Inversion Recovery':
            maxRiseTime = self.TR - (self.boards['slice']['objects']['spoiler']['time'][-1] - self.boards['RF']['objects']['inversion']['time'][0])
            maxAmp = self.maxSlew * maxRiseTime
            minThks.append(self.boards['RF']['objects']['inversion']['FWHM_f'] / (maxAmp * GYRO))
        else:
            maxRiseTime = self.TR - (self.boards['slice']['objects']['spoiler']['time'][-1] - self.boards['RF']['objects']['excitation']['time'][0])
            maxAmp = self.maxSlew * maxRiseTime
            minThks.append(self.boards['RF']['objects']['excitation']['FWHM_f'] / (maxAmp * GYRO))
        
        # See readouts.tex for formulae
        s = self.maxSlew
        d = self.boards['RF']['objects']['excitation']['dur_f']
        if isGradientEcho(self.sequence): # Constraint due to slice rephaser
            t = self.boards['ADC']['objects']['samplings'][0][0]['time'][0]
            h = s * (t - np.sqrt(t**2/2 + d**2/8))
            h = min(h, self.maxAmp)
            A = d * (np.sqrt((d*s+2*h)**2 - 8*h*(h-s*(t-d/2))) - d*s - 2*h) / 2
        else: # Spin echo: Constraint due to slice rephaser and refocusing slice select rampup
            t = self.boards['RF']['objects']['refocusing'][0]['time'][0]
            h = s * (np.sqrt(2*(d + 2*t)**2 - 4*d**2) - d - 2*t) / 4
            h = min(h, self.maxAmp)
            A = (np.sqrt((d*(d*s + 4*h))**2 - 4*d**2*h*(d*s + 2*h - 2*s*t)) - d*(d*s + 4*h)) / 2
        Be = self.boards['RF']['objects']['excitation']['FWHM_f']
        minThks.append(Be * d / (GYRO * A)) # mm
        
        self.param.sliceThickness.bounds = getBounds(max(minThks), 10., self.sliceThickness)


    def updateTurboFactorBounds(self):
        if not self.EPIfactor%2: # if even EPIfactor
            self.param.turboFactor.bounds = (1, 1)
            self.param.turboFactor.constant = True
        else:
            self.param.turboFactor.bounds = (1, 64)
            self.param.turboFactor.constant = False


    def updateEPIfactorObjects(self):
        if self.turboFactor > 1: # only odd EPIfactor for GRASE
            self.param.EPIfactor.objects = [v for v in EPIfactorValues if v%2]
        else:
            self.param.EPIfactor.objects = EPIfactorValues


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
        self.phaseDir = 1 - self.freqDir
        if self.freqDir==0:
            self.matrix.reverse()
            self.FOV.reverse()

        self.num_shots = int(np.ceil(self.matrix[self.phaseDir] * (1 + self.phaseOversampling / 100) * self.partialFourier / self.turboFactor / self.EPIfactor))
        self.num_measured_lines = self.num_shots * self.turboFactor * self.EPIfactor
        self.oversampledPartialMatrix = self.num_measured_lines # Needs to be modified for parallel imaging
        
        self.oversampledMatrix = self.matrix.copy() # account for oversampling
        # phase encoding direction: oversampling may be higher than prescribed since num_shots must be integer
        self.oversampledMatrix[self.phaseDir] = int(np.ceil(self.oversampledPartialMatrix / self.partialFourier))
        # frequency encoding direction: at least Nyquist sampling wrt phantom
        if self.FOV[self.freqDir] < self.phantom['FOV'][self.freqDir]:
            self.oversampledMatrix[self.freqDir] = int(np.ceil(self.phantom['FOV'][self.freqDir] * self.matrix[self.freqDir] / self.FOV[self.freqDir]))
        
        # Full k-space:
        self.kAxes = [getKaxis(self.oversampledMatrix[dim], self.FOV[dim]/self.matrix[dim]) for dim in range(len(self.matrix))]
        # Undersample by partial Fourier:
        self.kAxes[self.phaseDir] = self.kAxes[self.phaseDir][:self.oversampledPartialMatrix]
        self.plainKspaceComps = resampleKspace(self.phantom, self.kAxes)
        self.sampledMatrix = [len(ax) for ax in self.kAxes]
        assert(self.sampledMatrix[self.freqDir] == self.oversampledMatrix[self.freqDir])
        assert(self.sampledMatrix[self.phaseDir] == self.num_measured_lines)
        
        # Lorenzian line shape to mimic slice thickness
        blurFactor = .5
        sliceThicknessFilter = self.sliceThickness * np.outer(*[np.exp(-blurFactor * self.sliceThickness * np.abs(ax)) for ax in self.kAxes])
        for tissue in self.plainKspaceComps:
            self.plainKspaceComps[tissue] *= sliceThicknessFilter
        
        # signal for SNR calculation
        self.signal = np.sqrt(self.oversampledMatrix[self.freqDir] * self.num_measured_lines) * self.sliceThickness * np.prod(self.FOV)/np.prod(self.matrix)


    def updateSamplingTime(self):
        self.samplingTime = self.kAxes[self.freqDir] * self.FOV[self.freqDir] / self.matrix[self.freqDir] / self.pixelBandWidth * 1e3 # msec
        self.noiseStd = self.noiseGain / np.sqrt(np.diff(self.samplingTime[:2]) * self.NSA) / self.fieldStrength
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
            if tissue==self.phantom['referenceTissue']:
                self.decayedSignal = self.signal * np.take(T2w, np.argmin(np.abs(self.kAxes[self.freqDir])), axis=self.freqDir)[0]
    
    
    def simulateNoise(self):
        self.noise = np.random.normal(0, self.noiseStd, self.sampledMatrix) + 1j * np.random.normal(0, self.noiseStd, self.sampledMatrix)


    def updatePDandT1w(self):
        self.PDandT1w = {component: getPDandT1w(component, self.sequence, self.TR, self.TE, self.TI, self.FA, self.fieldStrength) for component in self.tissues.union(set(FATRESONANCES.keys()))}


    def setReferenceSNR(self, event=None):
        self.referenceSNR = self.SNR
        self.setRelativeSNR()


    def setRelativeSNR(self):
        self.relativeSNR = self.SNR / self.referenceSNR * 100


    def updateSNR(self, signal):
        self.SNR = signal / self.noiseStd[0]
        self.setRelativeSNR()
    

    def updateScantime(self):
        self.scantime = self.num_shots * self.NSA * self.TR * 1e-3 # scantime in seconds


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
        self.updateSNR(self.decayedSignal * np.abs(self.PDandT1w[self.phantom['referenceTissue']]))
        self.updateScantime()

    
    def partialFourierRecon(self):
        # Just zerofill for now
        nZeroes = self.oversampledMatrix[self.phaseDir] - self.oversampledPartialMatrix
        shape = tuple(nZeroes if dim==self.phaseDir else n for dim, n in enumerate(self.kspace.shape))
        self.kspace = np.append(self.kspace, np.zeros(shape), axis=self.phaseDir)
    
    
    def zerofill(self):
        self.reconMatrix = [self.reconMatrixP, self.reconMatrixF]
        if self.freqDir==0:
            self.reconMatrix.reverse()
        self.oversampledReconMatrix = self.reconMatrix.copy()
        for dim in range(len(self.oversampledReconMatrix)):
            self.oversampledReconMatrix[dim] = int(np.round(self.reconMatrix[dim] * self.oversampledMatrix[dim] / self.matrix[dim]))
        self.zerofilledkspace = zerofill(np.fft.ifftshift(self.kspace), self.oversampledReconMatrix)
    
    
    def reconstruct(self):
        pixelShifts = [.0] * len(self.reconMatrix) # pixel shift
        for dim in range(len(pixelShifts)):
            if not self.oversampledReconMatrix[dim]%2:
                pixelShifts[dim] += 1/2 # half pixel shift for even matrixsize due to fft
            if (self.oversampledReconMatrix[dim] - self.reconMatrix[dim])%2:
                pixelShifts[dim] += 1/2 # half pixel shift due to cropping an odd number of pixels in image space
        halfPixelShift = getPixelShiftMatrix(self.oversampledReconMatrix, pixelShifts)
        sampleShifts = [0. if self.oversampledMatrix[dim]%2 else .5 for dim in range(len(self.oversampledMatrix))]
        halfSampleShift = getPixelShiftMatrix(self.oversampledReconMatrix, sampleShifts)
        
        kspace = self.zerofilledkspace * halfPixelShift
        self.imageArray = np.fft.ifft2(kspace) * halfSampleShift
        self.imageArray = crop(np.fft.fftshift(self.imageArray), self.reconMatrix)
    
    
    def setupExcitation(self):
        FA = self.FA if isGradientEcho(self.sequence) else 90.
        self.boards['RF']['objects']['excitation'] = sequence.getRF(flipAngle=FA, time=0., dur=3., shape='hammingSinc',  name='excitation')
        for f in [self.setupSliceSelection, self.placeFatSat, self.renderRFBoard, self.updateMinTE, self.updateBWbounds, self.updateMatrixFbounds, self.updateFOVFbounds, self.updateMatrixPbounds, self.updateFOVPbounds, self.updateSliceThicknessBounds]:
            self.sequencePipeline.add(f)


    def setupRefocusing(self):
        self.boards['RF']['objects']['refocusing'] = []
        if not isGradientEcho(self.sequence):
            for rf_echo in range(self.turboFactor):
                self.boards['RF']['objects']['refocusing'].append(sequence.getRF(flipAngle=180., dur=3., shape='hammingSinc',  name='refocusing {}'.format(rf_echo)))
            self.sequencePipeline.add(self.placeRefocusing)
        for f in [self.setupSliceSelection, self.renderRFBoard, self.updateMinTE, self.updateMatrixPbounds, self.updateFOVPbounds, self.updateSliceThicknessBounds]:
            self.sequencePipeline.add(f)


    def setupInversion(self):
        if self.sequence=='Inversion Recovery':
            self.boards['RF']['objects']['inversion'] = sequence.getRF(flipAngle=180., dur=3., shape='hammingSinc',  name='inversion')
            self.sequencePipeline.add(self.placeInversion)
        elif 'inversion' in self.boards['RF']['objects']:
            del self.boards['RF']['objects']['inversion']
        for f in [self.setupSliceSelection, self.renderRFBoard, self.updateMaxTI, self.updateSliceThicknessBounds]:
            self.sequencePipeline.add(f)
    
    
    def setupFatSat(self):
        if self.FatSat:
            self.boards['RF']['objects']['fatsat'] = sequence.getRF(flipAngle=90, time=0., dur=30./self.fieldStrength, shape='hammingSinc',  name='FatSat')
            spoilerArea = 30. # uTs/m
            self.boards['slice']['objects']['fatsat spoiler'] = sequence.getGradient('slice', totalArea=spoilerArea, name='FatSat spoiler')
        elif 'fatsat' in self.boards['RF']['objects']:
            del self.boards['RF']['objects']['fatsat']
            del self.boards['slice']['objects']['fatsat spoiler']
        self.sequencePipeline.add(self.placeFatSat)
    
    
    def setupSliceSelection(self):
        flatDur = self.boards['RF']['objects']['excitation']['dur_f']
        amp = self.boards['RF']['objects']['excitation']['FWHM_f'] / (self.sliceThickness * GYRO)
        sliceSelectExcitation = sequence.getGradient('slice', 0., maxAmp=amp, flatDur=flatDur, name='slice select excitation')
        sliceRephaserArea = -sliceSelectExcitation['area_f']/2
        sliceSelectRephaser = sequence.getGradient('slice', totalArea=sliceRephaserArea, name='slice select rephaser')
        rephaserTime = (sliceSelectExcitation['dur_f'] + sliceSelectRephaser['dur_f']) / 2
        sequence.moveWaveform(sliceSelectRephaser, rephaserTime)
        self.boards['slice']['objects']['slice select excitation'] = sliceSelectExcitation
        self.boards['slice']['objects']['slice select rephaser'] = sliceSelectRephaser

        if 'refocusing' in self.boards['RF']['objects']:
            flatDur = self.boards['RF']['objects']['refocusing'][0]['dur_f']
            amp = self.boards['RF']['objects']['refocusing'][0]['FWHM_f'] / (self.sliceThickness * GYRO)
            self.boards['slice']['objects']['slice select refocusing'] = []
            for rf_echo in range(self.turboFactor):
                self.boards['slice']['objects']['slice select refocusing'].append(sequence.getGradient('slice', maxAmp=amp, flatDur=flatDur, name='slice select refocusing'))
            self.sequencePipeline.add(self.placeRefocusing)
        elif 'slice select refocusing' in self.boards['slice']['objects']:
            del self.boards['slice']['objects']['slice select refocusing']
            
        if 'inversion' in self.boards['RF']['objects']:
            flatDur = self.boards['RF']['objects']['inversion']['dur_f']
            amp = self.boards['RF']['objects']['inversion']['FWHM_f'] / (self.inversionThkFactor * self.sliceThickness * GYRO)
            self.boards['slice']['objects']['slice select inversion'] = sequence.getGradient('slice', maxAmp=amp, flatDur=flatDur, name='slice select inversion')

            spoilerArea = 30. # uTs/m
            self.boards['slice']['objects']['inversion spoiler'] = sequence.getGradient('slice', totalArea=spoilerArea, name='inversion spoiler')
            self.sequencePipeline.add(self.placeInversion)
        elif 'slice select inversion' in self.boards['slice']['objects']:
            del self.boards['slice']['objects']['slice select inversion']
            del self.boards['slice']['objects']['inversion spoiler']
        
        for f in [self.renderFrequencyBoard, self.renderPhaseBoard, self.renderSliceBoard, self.renderRFBoard, self.updateMinTE, self.updateMaxTI, self.updateMinTR, self.updateBWbounds]:
            self.sequencePipeline.add(f)
    

    def setupReadouts(self):
        flatArea = self.matrixF / (self.FOVF/1e3 * GYRO) # uTs/m
        amp = self.pixelBandWidth * self.matrixF / (self.FOVF * GYRO) # mT/m
        self.boards['frequency']['objects']['readouts'] = []
        self.boards['ADC']['objects']['samplings'] = []
        for rf_echo in range(self.turboFactor):
            gr_echoes = []
            samplings = []
            for gr_echo in range(self.EPIfactor):
                readout = sequence.getGradient('frequency', maxAmp=amp, flatArea=flatArea, name='readout r{} g{}'.format(rf_echo, gr_echo))
                gr_echoes.append(readout)
                adc = sequence.getADC(dur=readout['flatDur_f'], name='sampling r{} g{}'.format(rf_echo, gr_echo))
                samplings.append(adc)
            self.boards['frequency']['objects']['readouts'].append(gr_echoes)
            self.boards['ADC']['objects']['samplings'].append(samplings)
        prephaser = sequence.getGradient('frequency', totalArea=readout['area_f']/2, name='read prephaser')        
        self.boards['frequency']['objects']['read prephaser'] = prephaser
        self.sequencePipeline.add(self.placeReadouts)
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
        if not isGradientEcho(self.sequence):
            if self.kspaceOrder=='early echo':
                self.rf_echo_spacing = self.TE
            elif self.kspaceOrder=='center echo':
                self.rf_echo_spacing = 2 * self.TE / (self.turboFactor + 1)
            elif self.kspaceOrder=='late echo':
                self.rf_echo_spacing = self.TE / self.turboFactor
            
            for rf_echo in range(self.turboFactor):
                if self.kspaceOrder=='early echo':
                    pos = self.TE + (rf_echo - 1/2) * self.rf_echo_spacing
                elif self.kspaceOrder=='center echo':
                    pos = self.TE + (rf_echo - self.turboFactor / 2) * self.rf_echo_spacing
                elif self.kspaceOrder=='late echo':
                    pos = self.TE - (rf_echo+ 1/2) * self.rf_echo_spacing
                sequence.moveWaveform(self.boards['RF']['objects']['refocusing'][rf_echo], pos)
                sequence.moveWaveform(self.boards['slice']['objects']['slice select refocusing'][rf_echo], pos)
            self.sequencePipeline.add(self.renderRFBoard)
            self.sequencePipeline.add(self.renderSliceBoard)
            self.sequencePipeline.add(self.updateBWbounds)
    

    def placeInversion(self):
        for board, name, renderer in [('RF', 'inversion', self.renderRFBoard), ('slice', 'slice select inversion', self.renderSliceBoard)]:
            if name in self.boards[board]['objects']:
                sequence.moveWaveform(self.boards[board]['objects'][name], -self.TI)
                self.sequencePipeline.add(renderer)
        if 'inversion spoiler' in self.boards['slice']['objects']:
            spoilerTime = self.boards['RF']['objects']['inversion']['time'][-1] + self.boards['slice']['objects']['inversion spoiler']['dur_f']/2
            sequence.moveWaveform(self.boards['slice']['objects']['inversion spoiler'], spoilerTime)
        for f in [self.updateMinTR, self.updateBWbounds, self.updateSliceThicknessBounds, self.renderFrequencyBoard, self.renderPhaseBoard, self.renderSliceBoard, self.renderRFBoard]:
            self.sequencePipeline.add(f)
    
    
    def placeFatSat(self):
        if 'fatsat' in self.boards['RF']['objects']:
            t = self.boards['slice']['objects']['slice select excitation']['time'][0] - self.boards['slice']['objects']['fatsat spoiler']['dur_f']/2
            sequence.moveWaveform(self.boards['slice']['objects']['fatsat spoiler'], t)
            t -= (self.boards['slice']['objects']['fatsat spoiler']['dur_f'] + self.boards['RF']['objects']['fatsat']['dur_f']) / 2
            sequence.moveWaveform(self.boards['RF']['objects']['fatsat'], t)
        for f in [self.updateMinTR, self.renderRFBoard, self.renderFrequencyBoard, self.renderPhaseBoard, self.renderSliceBoard]:
            self.sequencePipeline.add(f)
    

    def placeReadouts(self):
        for rf_echo in range(self.turboFactor):
            for gr_echo in range(self.EPIfactor):
                if self.kspaceOrder=='early echo':
                    pos = self.TE + rf_echo * self.rf_echo_spacing
                elif self.kspaceOrder=='center echo':
                    pos = self.TE + (rf_echo - (self.turboFactor-1) / 2) * self.rf_echo_spacing
                elif self.kspaceOrder=='late echo':
                    pos = self.TE - rf_echo * self.rf_echo_spacing
                for object in [self.boards['frequency']['objects']['readouts'], self.boards['ADC']['objects']['samplings']]:
                    sequence.moveWaveform(object[rf_echo][gr_echo], pos)
        if isGradientEcho(self.sequence):
            if self.boards['frequency']['objects']['read prephaser']['area_f'] > 0:
                sequence.rescaleGradient(self.boards['frequency']['objects']['read prephaser'], -1)
            prephaseTime = self.TE - sum([grad['dur_f'] for grad in [self.boards['frequency']['objects']['read prephaser'], self.boards['frequency']['objects']['readouts'][0][0]]])/2
        else:
            if self.boards['frequency']['objects']['read prephaser']['area_f'] < 0:
                sequence.rescaleGradient(self.boards['frequency']['objects']['read prephaser'], -1)
            prephaseTime = sum([self.boards[b]['objects'][name]['dur_f'] for (b, name) in [('RF', 'excitation'), ('frequency', 'read prephaser')]])/2
        sequence.moveWaveform(self.boards['frequency']['objects']['read prephaser'], prephaseTime)
        self.sequencePipeline.add(self.placePhaser)
        self.sequencePipeline.add(self.placeSpoiler)
        self.sequencePipeline.add(self.renderFrequencyBoard)
    
    
    def placePhaser(self):
        phaserDur = self.boards['phase']['objects']['phase encode']['dur_f']
        if isGradientEcho(self.sequence):
            phaserTime = (self.boards['RF']['objects']['excitation']['dur_f'] + phaserDur)/2
        else:
            phaserTime = self.TE - (self.boards['frequency']['objects']['readouts'][0][0]['flatDur_f'] + phaserDur)/2
        sequence.moveWaveform(self.boards['phase']['objects']['phase encode'], phaserTime)
        self.sequencePipeline.add(self.renderPhaseBoard)
    

    def placeSpoiler(self):
        spoilerTime = self.boards['frequency']['objects']['readouts'][0][0]['center_f'] + (self.boards['frequency']['objects']['readouts'][0][0]['flatDur_f'] + self.boards['slice']['objects']['spoiler']['dur_f']) / 2
        sequence.moveWaveform(self.boards['slice']['objects']['spoiler'], spoilerTime)
        for f in [self.renderSliceBoard, self.updateMinTR, self.updateSliceThicknessBounds]:
            self.sequencePipeline.add(f)


    def renderPolygons(self, board):
        objects = flatten_dicts(self.boards[board]['objects'].values())
        self.boardPlots[board]['area'] = hv.Area(sequence.accumulateWaveforms(objects, board), self.timeDim, self.boards[board]['dim']).opts(color=self.boards[board]['color'])
        self.boards[board]['attributes'] = []
        if self.boards[board]['objects']:
            self.boards[board]['attributes'] += [attr for attr in objects[0].keys() if attr not in ['time', board] and '_f' not in attr]
            hover = HoverTool(tooltips=[(attr, '@{}'.format(attr)) for attr in self.boards[board]['attributes']], attachment='below')
            self.boardPlots[board]['polygons'] = hv.Polygons(objects, kdims=[self.timeDim, self.boards[board]['dim']], vdims=self.boards[board]['attributes']).opts(tools=[hover], cmap=[self.boards[board]['color']], hooks=[hideframe_hook, partial(bounds_hook, xbounds=(-19000, 19000))])
    
    
    def renderTRspan(self, board):
        t0 = self.getSeqStart()
        self.boardPlots[board]['TRspan'] = hv.VSpan(-20000, t0, kdims=[self.timeDim, self.boards[board]['dim']]).opts(color='gray', fill_alpha=.3)
        self.boardPlots[board]['TRspan'] *= hv.VSpan(t0 + self.TR, 20000, kdims=[self.timeDim, self.boards[board]['dim']]).opts(color='gray', fill_alpha=.3)
    

    def renderFrequencyBoard(self):
        self.renderPolygons('frequency')
        adc_objects = flatten_dicts(self.boards['ADC']['objects'].values())
        self.boardPlots['frequency']['ADC'] = hv.Overlay([hv.VSpan(obj['time'][0], obj['time'][-1], kdims=[self.timeDim, self.boards['frequency']['dim']]) for obj in adc_objects])
        self.renderTRspan('frequency')
    

    def renderPhaseBoard(self):
        self.renderPolygons('phase')
        self.renderTRspan('phase')
    

    def renderSliceBoard(self):
        self.renderPolygons('slice')
        self.renderTRspan('slice')


    def renderRFBoard(self):
        self.renderPolygons('RF')
        self.renderTRspan('RF')
    
    
    @param.depends('object', 'fieldStrength', 'sequence', 'FatSat', 'TR', 'TE', 'FA', 'TI', 'FOVF', 'FOVP', 'phaseOversampling', 'matrixF', 'matrixP', 'reconMatrixF', 'reconMatrixP', 'sliceThickness', 'frequencyDirection', 'pixelBandWidth', 'NSA', 'kspaceOrder', 'partialFourier', 'turboFactor', 'EPIfactor')
    def getKspace(self):
        if self.render:
            self.runReconPipeline()
            kAxes = []
            for dim in range(2):
                kAxes.append(getKaxis(self.oversampledReconMatrix[dim], self.FOV[dim]/self.reconMatrix[dim]))
                # half-sample shift axis when odd number of zeroes:
                if (self.oversampledReconMatrix[dim]-self.oversampledMatrix[dim])%2:
                    shift = self.reconMatrix[dim] / (2 * self.oversampledReconMatrix[dim] * self.FOV[dim])
                    kAxes[-1] += shift * (-1)**(self.oversampledMatrix[dim]%2)
            ksp = xr.DataArray(
                np.abs(np.fft.fftshift(self.zerofilledkspace))**.2, 
                dims=('ky', 'kx'),
                coords={'kx': kAxes[1], 'ky': kAxes[0]}
            )
            ksp.kx.attrs['units'] = ksp.ky.attrs['units'] = '1/mm'
            self.kspaceimage = hv.Image(ksp, vdims=['magnitude'])
        return self.kspaceimage


    def getFOVbox(self):
        FOV = (self.FOVP, self.FOVF)
        acqFOV = (self.FOVP * self.oversampledMatrix[self.phaseDir] / self.matrix[self.phaseDir], self.FOVF)
        if self.frequencyDirection == 'left-right':
            FOV, acqFOV = FOV.reverse(), acqFOV.reverse()
        return hv.Box(0, 0, acqFOV).opts(color='lightblue') * hv.Box(0, 0, FOV).opts(color='yellow')


    @param.depends('object', 'fieldStrength', 'sequence', 'FatSat', 'TR', 'TE', 'FA', 'TI', 'FOVF', 'FOVP', 'phaseOversampling', 'matrixF', 'matrixP', 'reconMatrixF', 'reconMatrixP', 'sliceThickness', 'frequencyDirection', 'pixelBandWidth', 'NSA', 'kspaceOrder', 'partialFourier', 'turboFactor', 'EPIfactor', 'showFOV')
    def getImage(self):
        if self.render:
            self.runReconPipeline()
            iAxes = [(np.arange(self.reconMatrix[dim]) - (self.reconMatrix[dim]-1)/2) / self.reconMatrix[dim] * self.FOV[dim] for dim in range(2)]
            img = xr.DataArray(
                np.abs(self.imageArray), 
                dims=('y', 'x'),
                coords={'x': iAxes[1], 'y': iAxes[0][::-1]}
            )
            img.x.attrs['units'] = img.y.attrs['units'] = 'mm'
            self.image = hv.Overlay([hv.Image(img, vdims=['magnitude'])])
        
        if self.showFOV:
            self.image *= self.getFOVbox()
        return self.image


    @param.depends('sequence', 'FatSat', 'TR', 'TE', 'FA', 'TI', 'FOVF', 'FOVP', 'matrixF', 'matrixP', 'sliceThickness', 'pixelBandWidth', 'kspaceOrder', 'partialFourier', 'turboFactor', 'EPIfactor')
    def getSequencePlot(self):
        if self.render:
            self.runSequencePipeline()
            self.seqPlot = hv.Layout(list([hv.Overlay(list(boardPlot.values())).opts(width=1700, height=120, border=0, xaxis='bottom' if n==len(self.boardPlots)-1 else None) for n, boardPlot in enumerate(self.boardPlots.values())])).cols(1).options(toolbar='below')
        return self.seqPlot


def hideShowButtonCallback(pane, event):
    if 'Show' in event.obj.name:
        pane.visible = True
        event.obj.name = event.obj.name.replace('Show', 'Hide')
    elif 'Hide' in event.obj.name:
        pane.visible = False
        event.obj.name = event.obj.name.replace('Hide', 'Show')


def infoNumber(name, value, format):
    return pn.indicators.Number(default_color='white', name=name, format=format, font_size='12pt', title_size='12pt', value=value)


def getApp():
    explorer = MRIsimulator(name='')
    title = '# SpinSight MRI simulator'
    author = '*Written by [Johan Berglund](mailto:johan.berglund@akademiska.se), Ph.D.*'
    settingsParams = pn.panel(explorer.param, parameters=['object', 'fieldStrength', 'parameterStyle'], name='Settings')
    contrastParams = pn.panel(explorer.param, parameters=['FatSat', 'TR', 'TE', 'FA', 'TI'], widgets={'TR': pn.widgets.DiscreteSlider, 'TE': pn.widgets.DiscreteSlider, 'TI': pn.widgets.DiscreteSlider}, name='Contrast')
    geometryParams = pn.panel(explorer.param, parameters=['frequencyDirection', 'FOVF', 'FOVP', 'phaseOversampling', 'voxelF', 'voxelP', 'matrixF', 'matrixP', 'reconVoxelF', 'reconVoxelP', 'reconMatrixF', 'reconMatrixP', 'sliceThickness'], widgets={'matrixF': pn.widgets.DiscreteSlider, 'matrixP': pn.widgets.DiscreteSlider, 'voxelF': pn.widgets.DiscreteSlider, 'voxelP': pn.widgets.DiscreteSlider, 'reconVoxelF': pn.widgets.DiscreteSlider, 'reconVoxelP': pn.widgets.DiscreteSlider}, name='Geometry')
    sequenceParams = pn.panel(explorer.param, parameters=['sequence', 'pixelBandWidth', 'FOVbandwidth', 'FWshift', 'NSA', 'kspaceOrder', 'partialFourier', 'turboFactor', 'EPIfactor'], widgets={'EPIfactor': pn.widgets.DiscreteSlider}, name='Sequence')
    
    infoPane = pn.Row(infoNumber(name='Relative SNR', format='{value:.0f}%', value=explorer.param.relativeSNR),
                      infoNumber(name='Scan time', format=('{value:.1f} sec'), value=explorer.param.scantime),
                      infoNumber(name='Fat/water shift', format='{value:.2f} pixels', value=explorer.param.FWshift),
                      infoNumber(name='Bandwidth', format='{value:.0f} Hz/pixel', value=explorer.param.pixelBandWidth))

    dmapKspace = pn.Row(hv.DynamicMap(explorer.getKspace), visible=False)
    dmapMRimage = hv.DynamicMap(explorer.getImage)
    dmapSequence = pn.Row(hv.DynamicMap(explorer.getSequencePlot), visible=False)
    sequenceButton = pn.widgets.Button(name='Show sequence')
    sequenceButton.on_click(partial(hideShowButtonCallback, dmapSequence))
    kSpaceButton = pn.widgets.Button(name='Show k-space')
    kSpaceButton.on_click(partial(hideShowButtonCallback, dmapKspace))
    resetSNRbutton = pn.widgets.Button(name='Set reference SNR')
    resetSNRbutton.on_click(explorer.setReferenceSNR)
    dashboard = pn.Column(
        pn.Row(
            pn.Column(
                pn.pane.Markdown(title), 
                pn.Row(
                    pn.Column(
                        settingsParams, 
                        pn.Row(sequenceButton, kSpaceButton), 
                        sequenceParams
                    ), 
                    pn.Column(
                        contrastParams,
                        geometryParams
                    )
                )
            ), 
            pn.Column(
                dmapMRimage, 
                pn.Column(
                    pn.Row(resetSNRbutton, explorer.param.showFOV), 
                    infoPane
                )
            ), 
            dmapKspace
        ), 
        dmapSequence, 
        pn.pane.Markdown(author)
    )
    return dashboard