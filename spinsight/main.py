import holoviews as hv
from holoviews import streams
import panel as pn
import param
import numpy as np
import math
import finufft
import xml.etree.ElementTree as ET
from pathlib import Path
import toml
import re
import xarray as xr
from spinsight import constants
from spinsight import sequence
from bokeh.models import HoverTool, CustomJS, ColumnDataSource
from functools import partial
from tqdm import tqdm

hv.extension('bokeh')


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

TRAJECTORIES = ['Cartesian', 'Radial']


def pixelBW2shift(pixelBW, B0=1.5):
    ''' Get fat/water chemical shift [pixels] from pixel bandwidth [Hz/pixel] and B0 [T]'''
    return np.abs(FATRESONANCES['Fat2']['shift'] * constants.GYRO * B0 / pixelBW)


def shift2pixelBW(shift, B0=1.5):
    ''' Get pixel bandwidth [Hz/pixel] from fat/water chemical shift [pixels] and B0 [T]'''
    return np.abs(FATRESONANCES['Fat2']['shift'] * constants.GYRO * B0 / shift)


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
    match = re.search(r'translate\((-?\d+.\d+)(px|%), (-?\d+.\d+)(px|%)\)', transformString)
    translation = (float(match.group(1)), float(match.group(3))) if match else (0, 0)

    match = re.search(r'rotate\((-?\d+.\d+)deg\)', transformString)
    rotation = float(match.group(1)) if match else 0
    
    match = re.search(r'scale\((-?\d+.\d+)\)', transformString)
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


def resampleKspaceCartesian(phantom, kAxes):
    kspace = {tissue: phantom['kspace'][tissue] for tissue in phantom['kspace']}
    for dim in range(len(kAxes)):
        sinc = np.sinc((np.tile(kAxes[dim], (len(phantom['kAxes'][dim]), 1)) - np.tile(phantom['kAxes'][dim][:, np.newaxis], (1, len(kAxes[dim])))) * phantom['FOV'][dim])
        for tissue in phantom['kspace']:
            kspace[tissue] = np.moveaxis(np.tensordot(kspace[tissue], sinc, axes=(dim, 0)), -1, dim)
    return kspace


def resampleKspace(phantom, kSamples):
    kspace = {}
    for tissue in phantom['kspace']:
        img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(phantom['kspace'][tissue])))
        kx = kSamples[..., 0].flatten() * 2 * np.pi * phantom['FOV'][0] / phantom['matrix'][0]
        ky = kSamples[..., 1].flatten() * 2 * np.pi * phantom['FOV'][1] / phantom['matrix'][1]
        kspace[tissue] = finufft.nufft2d2(kx, ky, img).reshape(kSamples.shape[:-1])
    return kspace


def homodyneWeights(N, nBlank, dim):
    # create homodyne ramp filter of length N with nBlank unsampled lines
    W = np.ones((N,))
    W[nBlank-1:-nBlank+1] = np.linspace(1,0, N-2*(nBlank-1))
    W[-nBlank:] = 0
    shape = (N, 1) if dim==0 else (1, N)
    return W.reshape(shape)


def radialTukey(alpha, matrix):
    kx = np.linspace(-1, 1, matrix[0]+2)[1:-1]
    ky = np.linspace(-1, 1, matrix[1]+2)[1:-1]
    kky, kkx = np.meshgrid(ky, kx)
    k = (1-np.sqrt(kkx**2 + kky**2))/alpha
    k[k>1] = 1
    k[k<0] = 0
    return np.sin(np.pi*k/2)**2


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


def getT2w(component, timeAfterExcitation, timeRelativeInphase, B0):
    T2 = TISSUES[component]['T2'][B0] if 'Fat' not in component else FATRESONANCES[component]['T2'][B0]
    T2prim = 35. # ad hoc value [msec]
    E2 = np.exp(-np.abs(timeAfterExcitation)/T2)
    E2prim = np.exp(-np.abs(timeRelativeInphase)/T2prim)
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


def get_segment_order(N, Nsym, c):
    '''Returns the temporal order in which to read k-space segments for a spin-echo train

    Args:
        N: number of segments, i.e. echo train length
        Nsym: number of segments symmetric about the center of k-space
        c: index of spin echo where (the first) centermost k-space segment is read

    Returns:
        Segment indices as a temporally ordered list
    '''

    split_center = not(Nsym % 2) # k-space center is between two segments

    if c >= N-split_center:
            raise ValueError('The spin echo index of (the first) centermost k-space segment is too high')
    elif c > N//2 - split_center:
        return get_segment_order(N, Nsym, N-1-c-split_center)[::-1]
    
    Ncon = min(2 * c + 1 + split_center, Nsym) # number of symmetric segments to be read consecutively
    Npivot = Nsym - Ncon # number of symmetric segments to be read in a pivoting fashion
    Nasym = N - Nsym # number of asymmetric segments
    linear_start = Nasym + Nsym//2 - split_center - c # start of consecutively read segments
    linear_end = N - Npivot//2 # end of consecutively read segments (+1)
    linear = list(range(linear_start, linear_end)) # consecutively read segments
    if linear_start==Nasym:
        linear.reverse()
    # segments read in a pivoting fashion:
    pivot = [val for pair in zip(range(linear_end, N), reversed(range(Nasym, linear_start))) for val in pair]
    tail = list(range(min(linear_start, Nasym)))[::-1] # remaining asymmetric segments
    segment_order = linear + pivot + tail
    return segment_order


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


def take_closest(objects, value):
    return min(objects, key=lambda x: abs(x-value))


def add_to_pipeline(pipeline, functions):
    pipeline.update({f: True for f in functions})


def unique_list(lst):
    return list(dict.fromkeys(lst))


TRvalues = unique_list([float('{:.2g}'.format(tr)) for tr in 10.**np.linspace(0, 4, 500)])
TEvalues = unique_list([float('{:.2g}'.format(te)) for te in 10.**np.linspace(0, 3, 500)])
TIvalues = unique_list([float('{:.2g}'.format(ti)) for ti in 10.**np.linspace(0, 4, 500)])
pBWvalues = unique_list([float('{:.3g}'.format(pbw)) for pbw in 10.**np.linspace(2.1, 3.3, 500)])
matrixValues = list(range(16, 600+1))
reconMatrixValues = list(range(16, 1200+1))
EPIfactorValues = list(range(1, 64+1))


class MRIsimulator(param.Parameterized):
    object = param.ObjectSelector(default='brain', label='Phantom object')
    fieldStrength = param.ObjectSelector(default=1.5, label='B0 field strength [T]')
    parameterStyle = param.ObjectSelector(default='Matrix and Pixel BW', label='Parameter Style')
    
    FatSat = param.Boolean(default=False, label='Fat saturation')
    TR = param.Selector(default=10000, label='TR [msec]')
    TE = param.Selector(default=10, label='TE [msec]')
    FA = param.Number(default=90.0, precedence=-1, label='Flip angle [°]')
    TI = param.Selector(default=40, precedence=-1, label='TI [msec]')
    
    trajectory = param.ObjectSelector(default=TRAJECTORIES[0], precedence=1, label='k-space trajectory')
    frequencyDirection = param.ObjectSelector(default=list(DIRECTIONS.keys())[0], precedence=1, label='Frequency encoding direction')
    FOVP = param.Number(default=240, precedence=2, label='FOV x [mm]')
    FOVF = param.Number(default=240, precedence=2, label='FOV y [mm]')
    phaseOversampling = param.Number(default=0, step=1., precedence=3, label='Phase oversampling [%]')
    voxelP = param.Selector(default=1.333, precedence=-4, label='Voxel size x [mm]')
    voxelF = param.Selector(default=1.333, precedence=-4, label='Voxel size y [mm]')
    matrixP = param.Selector(default=180, precedence=4, label='Acquisition matrix x')
    matrixF = param.Selector(default=180, precedence=4, label='Acquisition matrix y')
    reconVoxelP = param.Selector(default=0.666, precedence=-5, label='Reconstructed voxel size x [mm]')
    reconVoxelF = param.Selector(default=0.666, precedence=-5, label='Reconstructed voxel size y [mm]')
    reconMatrixP = param.Selector(default=360, precedence=5, label='Reconstruction matrix x')
    reconMatrixF = param.Selector(default=360, precedence=5, label='Reconstruction matrix y')
    sliceThickness = param.Number(default=3, precedence=6, label='Slice thickness [mm]')
    
    sequence = param.ObjectSelector(default=SEQUENCES[0], precedence=1, label='Pulse sequence')
    pixelBandWidth = param.Selector(default=pBWvalues[249], precedence=2, label='Pixel bandwidth [Hz]')
    FOVbandwidth = param.Selector(default=pixelBW2FOVBW(500, 180), precedence=-2, label='FOV bandwidth [±kHz]')
    FWshift = param.Selector(default=pixelBW2shift(500), precedence=-2, label='Fat/water shift [pixels]')
    NSA = param.Integer(default=1, precedence=3, label='NSA')
    partialFourier = param.Number(default=1, step=0.01, precedence=5, label='Partial Fourier factor')
    turboFactor = param.Integer(default=1, precedence=6, label='Turbo factor')
    EPIfactor = param.Selector(default=1, precedence=7, label='EPI factor')
    shot = param.Integer(default=1, label='Displayed shot')
    
    showFOV = param.Boolean(default=False, label='Show FOV')
    noiseGain = param.Number(default=3.)
    SNR = param.Number(label='SNR')
    referenceSNR = param.Number(default=1, label='Reference SNR')
    relativeSNR = param.Number(label='Relative SNR [%]')
    scantime = param.Number(label='Scan time [sec]')

    showProcessedKspace = param.Boolean(default=False, label='Show processed k-space')
    kspaceExponent = param.Number(default=0.2, step=.01, label='k-space exponent')
    homodyne = param.Boolean(default=True, precedence=1, label='Homodyne')
    doApodize = param.Boolean(default=True, precedence=2, label='Apodization')
    apodizationAlpha = param.Number(default=0.25, step=.01, precedence=3, label='Apodization alpha')
    doZerofill = param.Boolean(default=True, precedence=4, label='Zerofill')

    def __init__(self, **params):
        super().__init__(**params)

        self.init_bounds()

        def arrow(coords):
            angle = 0
            if len(coords)>1:
                angle = -np.degrees(math.atan2(coords[-1][0]-coords[-2][0], coords[-1][1]-coords[-2][1]))
            return hv.Curve(coords) * hv.Points(coords[-1]).opts(angle=angle, marker='triangle')
        
        arrowStream = streams.Stream.define('arrow', coords=[None])
        self.kLine = hv.DynamicMap(arrow, streams=[arrowStream()])

        self.hoverIndex = ColumnDataSource({'index': [], 'board': []})
        self.hoverIndex.on_change('data', self.update_k_line_coords)
        
        self.outbound_params = set()

        self.timeDim = hv.Dimension('time', label='time', unit='ms')

        self.boards = { 'frequency': {'dim': hv.Dimension('frequency', label='G read', unit='mT/m', range=(-30, 30)), 'color': 'cadetblue'}, 
                        'phase': {'dim': hv.Dimension('phase', label='G phase', unit='mT/m', range=(-30, 30)), 'color': 'cadetblue'}, 
                        'slice': {'dim': hv.Dimension('slice', label='G slice', unit='mT/m', range=(-30, 30)), 'color': 'cadetblue'}, 
                        'RF': {'dim': hv.Dimension('RF', label='RF', unit='μT', range=(-5, 25)), 'color': 'red'},
                        'signal': {'dim': hv.Dimension('signal', label='signal', unit='a.u.', range=(-1, 1)), 'color': 'orange'},
                        'ADC': {'dim': hv.Dimension('ADC', label='ADC', unit=''), 'color': 'peru'} }
        
        self.boardPlots = {board: {'hline': hv.HLine(0.0, kdims=[self.timeDim, self.boards[board]['dim']]).opts(tools=['xwheel_zoom', 'xpan', 'reset'], default_tools=[], active_tools=['xwheel_zoom', 'xpan'])} for board in self.boards if board != 'ADC'}

        hv.opts.defaults(hv.opts.Image(width=500, height=500, invert_yaxis=False, toolbar='below', cmap='gray', aspect='equal'))
        hv.opts.defaults(hv.opts.HLine(line_width=1.5, line_color='gray'))
        hv.opts.defaults(hv.opts.VSpan(color='orange', fill_alpha=.1, hover_fill_alpha=.8, default_tools=[]))
        hv.opts.defaults(hv.opts.Rectangles(color=self.boards['ADC']['color'], line_color=self.boards['ADC']['color'], fill_alpha=.1, line_alpha=.3, hover_fill_alpha=.8, default_tools=[]))
        hv.opts.defaults(hv.opts.Box(line_width=3))
        hv.opts.defaults(hv.opts.Area(fill_alpha=.5, line_width=1.5, line_color='gray', default_tools=[]))
        hv.opts.defaults(hv.opts.Polygons(line_width=1.5, fill_alpha=0, line_alpha=0, line_color='gray', selection_line_color='black', hover_fill_alpha=.8, hover_line_alpha=1, selection_fill_alpha=.8, selection_line_alpha=1, nonselection_line_alpha=0, default_tools=[]))
        hv.opts.defaults(hv.opts.Curve(line_width=5, line_color='peru'))
        hv.opts.defaults(hv.opts.Points(line_color=None, color='peru', size=15))

        self.maxAmp = 25. # mT/m
        self.maxSlew = 80. # T/m/s
        self.inversionThkFactor = 1.1 # make inversion slice 10% thicker

        for board in self.boards:
            self.boards[board]['objects'] = {}

        self.derivedParams = ['FOVbandwidth', 'FWshift', 'SNR', 'name', 'reconVoxelF', 'reconVoxelP', 'referenceSNR', 'relativeSNR', 'scantime', 'voxelF', 'voxelP']
        
        self.sequencePipeline = {f: True for f in [
            'setupExcitation', 
            'setupRefocusing',
            'setupInversion',
            'setupFatSat',
            'setupSliceSelection',
            'setupReadouts',
            'setupPhasers',
            'setupSpoiler',
            'placeRefocusing',
            'placeInversion',
            'placeFatSat',
            'placeReadouts',
            'placePhasers',
            'placeSpoiler',
            'updateMinTE',
            'updateMinTR',
            'updateMaxTE',
            'updateMaxTI',
            'updateBWbounds',
            'updateMatrixFbounds',
            'updateMatrixPbounds',
            'updateFOVFbounds', 
            'updateFOVPbounds',
            'updateSliceThicknessBounds'
            ]}
        
        self.acquisitionPipeline = {f: True for f in [
            'loadPhantom', 
            'sampleKspace', 
            'updateSamplingTime', 
            'modulateKspace', 
            'simulateNoise', 
            'updatePDandT1w', 
            'compileKspace'
        ]}

        self.sequencePlotPipeline = {f: True for f in [
            'renderFrequencyBoard', 
            'renderPhaseBoard', 
            'renderSliceBoard', 
            'renderRFBoard',
            'renderSignalBoard',
            'renderTRspan',
            'calculate_k_trajectory'
        ]}
        
        self.reconPipeline = {f: True for f in [
            'partialFourierRecon', 
            'apodization', 
            'zerofill', 
            'reconstruct', 
            'setReferenceSNR'
        ]}

        self._watch_reconMatrixF()
        self._watch_reconMatrixP()
        self._watch_matrixF()
        self._watch_matrixP()

        self.FOV = [None]*2
        self.matrix = [None]*2
        self.oversampledMatrix = [None]*2
        self.kAxes = [None]*2
        
        self.runSequencePipeline()


    def init_bounds(self):
        self.param.object.objects = PHANTOMS.keys()
        self.param.fieldStrength.objects=[1.5, 3.0]
        self.param.parameterStyle.objects=['Matrix and Pixel BW', 'Voxelsize and Fat/water shift', 'Matrix and FOV BW']
        self.param.TR.objects=TRvalues
        self.param.TE.objects=TEvalues
        self.param.FA.bounds=(1, 90.0)
        self.param.TI.objects=TIvalues
        self.param.frequencyDirection.objects=DIRECTIONS.keys()
        self.param.trajectory.objects=TRAJECTORIES
        self.param.FOVP.bounds=(100, 600)
        self.param.FOVF.bounds=(100, 600)
        self.param.phaseOversampling.bounds=(0, 100)
        self.param.matrixP.objects=matrixValues
        self.param.matrixF.objects=matrixValues
        self.param.reconMatrixP.objects=reconMatrixValues
        self.param.reconMatrixF.objects=reconMatrixValues
        self.param.sliceThickness.bounds=(0.5, 10)
        self.param.sequence.objects=SEQUENCES
        self.param.pixelBandWidth.objects=pBWvalues
        self.param.NSA.bounds=(1, 16)
        self.param.partialFourier.bounds=(.6, 1)
        self.param.turboFactor.bounds=(1, 64)
        self.param.EPIfactor.objects=EPIfactorValues
        self.param.shot.bounds=(1, 1)
        self.param.kspaceExponent.bounds=(0.1, 1)
        self.param.apodizationAlpha.bounds=(.01, 1)


    def runPipeline(self, pipeline):
        for f in pipeline:
            if pipeline[f]:
                self.__getattribute__(f)()
                pipeline[f] = False
    

    def runSequencePipeline(self):
        if not any(self.sequencePipeline.values()):
            return
        self.runPipeline(self.sequencePipeline)
        self.resolveConflicts()
    

    def runAcquisitionPipeline(self):
        if not any(self.acquisitionPipeline.values()):
            return
        self.runSequencePipeline()
        self.runPipeline(self.acquisitionPipeline)
    

    def runSequencePlotPipeline(self):
        if not any(self.sequencePlotPipeline.values()):
            return
        self.runSequencePipeline()
        self.runAcquisitionPipeline()
        self.runPipeline(self.sequencePlotPipeline)
    

    def runReconPipeline(self):
        if not any(self.reconPipeline.values()):
            return
        self.runSequencePipeline()
        self.runAcquisitionPipeline()
        self.runPipeline(self.reconPipeline)
    
    
    def getParams(self):
        return {param: self.__getattribute__(param) for param in self.param.values().keys() if param not in self.derivedParams}
    
    
    def setParams(self, settings):
        self.init_bounds()
        self.sequencePipeline = {f: True for f in self.sequencePipeline.keys()}
        self.param.update(settings)

    
    def setParamBounds(self, param, minval, maxval):
        curval = getattr(self, param.name)
        if curval < minval:
            print('Warning: trying to set {} bounds above current value ({} > {})'.format(param.name, minval, curval))
            minval = curval
        if curval > maxval:
            print('Warning: trying to set {} bounds below current value ({} < {})'.format(param.name, maxval, curval))
            maxval = curval
        param.bounds = (minval, maxval)
    

    def setParamDiscreteBounds(self, param, values, minval=None, maxval=None):
        curval = getattr(self, param.name)
        if minval is not None:
            values = [val for val in values if not val < minval]
        if maxval is not None:
            values = [val for val in values if not val > maxval]
        if curval not in values:
            print('Warning: {} current value {} is outside its new bounds [{}, {}]'.format(param.name, curval, minval, maxval))
            values = [curval] # necessary to avoid empty bounds
            self.outbound_params.add(param.name)
        elif param.name in self.outbound_params:
            print('Param {} no longer conflicting'.format(param.name))
            self.outbound_params.remove(param.name)
        param.objects = values
    

    @param.depends('object', watch=True)
    def _watch_object(self):
        with param.parameterized.batch_call_watchers(self):
            for f in self.acquisitionPipeline:
                self.acquisitionPipeline[f] = True
            for f in self.reconPipeline:
                self.reconPipeline[f] = True
            minFOV = PHANTOMS[self.object]['FOV']
            if self.frequencyDirection=='left-right':
                minFOV = minFOV.reverse()
            self.FOVF = max(self.FOVF, minFOV[0])
            self.FOVP = max(self.FOVP, minFOV[1])
    

    @param.depends('parameterStyle', watch=True)
    def _watch_parameterStyle(self):
        for par in [self.param.voxelF, self.param.voxelP, self.param.matrixF, self.param.matrixP, self.param.reconVoxelF, self.param.reconVoxelP, self.param.reconMatrixF, self.param.reconMatrixP, self.param.pixelBandWidth, self.param.FOVbandwidth, self.param.FWshift]:
            par.precedence = -1
        if self.parameterStyle == 'Voxelsize and Fat/water shift':
            self.param.voxelF.precedence = 4
            self.param.voxelP.precedence = 4
            self.param.reconVoxelF.precedence = 5
            self.param.reconVoxelP.precedence = 5
            self.param.FWshift.precedence = 2
            self._watch_reconMatrixF()
            self._watch_reconMatrixP()
            self._watch_pixelBandWidth()
        else:
            self.param.matrixF.precedence = 4
            self.param.matrixP.precedence = 4
            self.param.reconMatrixF.precedence = 5
            self.param.reconMatrixP.precedence = 5
            if self.parameterStyle == 'Matrix and Pixel BW':
                self.param.pixelBandWidth.precedence = 2
            elif self.parameterStyle == 'Matrix and FOV BW':
                self.param.FOVbandwidth.precedence = 2
                self.updateMatrixFbounds()


    @param.depends('FOVF', watch=True)
    def _watch_FOVF(self):
        with param.parameterized.batch_call_watchers(self):
            if self.parameterStyle=='Voxelsize and Fat/water shift':
                self.matrixF = take_closest(self.param.matrixF.objects, self.FOVF/self.voxelF)
                self.reconMatrixF = take_closest(self.param.reconMatrixF.objects, self.FOVF/self.reconVoxelF)
            self.updateFOVbandwidthObjects()
            self.updateVoxelFobjects()
            self.updateReconVoxelFobjects()
            self.voxelF = take_closest(self.param.voxelF.objects, self.FOVF/self.matrixF)
            self.reconVoxelF = take_closest(self.param.reconVoxelF.objects, self.FOVF/self.reconMatrixF)
            self.param.trigger('voxelF', 'reconVoxelF')
            add_to_pipeline(self.acquisitionPipeline, ['sampleKspace', 'updateSamplingTime', 'modulateKspace', 'simulateNoise', 'compileKspace'])
            add_to_pipeline(self.reconPipeline, ['partialFourierRecon', 'apodization', 'zerofill', 'reconstruct'])
            add_to_pipeline(self.sequencePipeline, ['setupReadouts', 'updateBWbounds', 'updateMatrixFbounds'])
    

    @param.depends('FOVP', watch=True)
    def _watch_FOVP(self):
        with param.parameterized.batch_call_watchers(self):
            if self.parameterStyle=='Voxelsize and Fat/water shift':
                self.matrixP = take_closest(self.param.matrixP.objects, self.FOVP/self.voxelP)
                self.reconMatrixP = take_closest(self.param.reconMatrixP.objects, self.FOVP/self.reconVoxelP)
            self.updateVoxelPobjects()
            self.updateReconVoxelPobjects()
            self.voxelP = take_closest(self.param.voxelP.objects, self.FOVP/self.matrixP)
            self.reconVoxelP = take_closest(self.param.reconVoxelP.objects, self.FOVP/self.reconMatrixP)
            self.param.trigger('voxelP', 'reconVoxelP')
            add_to_pipeline(self.acquisitionPipeline, ['sampleKspace', 'updateSamplingTime', 'modulateKspace', 'simulateNoise', 'compileKspace'])
            add_to_pipeline(self.reconPipeline, ['partialFourierRecon', 'apodization', 'zerofill', 'reconstruct'])
            add_to_pipeline(self.sequencePipeline, ['setupPhasers', 'updateMatrixPbounds'])


    @param.depends('phaseOversampling', watch=True)
    def _watch_phaseOversampling(self):
        self._watch_FOVP()


    @param.depends('matrixF', watch=True)
    def _watch_matrixF(self):
        with param.parameterized.batch_call_watchers(self):
            self.updateFOVbandwidthObjects()
            if self.parameterStyle == 'Matrix and FOV BW':
                self.pixelBandWidth = take_closest(self.param.pixelBandWidth.objects, FOVBW2pixelBW(self.FOVbandwidth, self.matrixF))
            else:
                self.FOVbandwidth = take_closest(self.param.FOVbandwidth.objects, pixelBW2FOVBW(self.pixelBandWidth, self.matrixF))
                self.param.trigger('FOVbandwidth')
            self.updateVoxelFobjects()
            self.voxelF = take_closest(self.param.voxelF.objects, self.FOVF/self.matrixF)
            add_to_pipeline(self.acquisitionPipeline, ['sampleKspace', 'updateSamplingTime', 'modulateKspace', 'simulateNoise', 'compileKspace'])
            add_to_pipeline(self.reconPipeline, ['partialFourierRecon', 'apodization', 'zerofill', 'reconstruct'])
            add_to_pipeline(self.sequencePipeline, ['setupReadouts', 'updateBWbounds', 'updateMatrixFbounds', 'updateFOVFbounds'])
            self.param.reconMatrixF.objects = [m for m in reconMatrixValues if m>=self.matrixF]
            self.updateReconVoxelFobjects()
            self.reconMatrixF = take_closest(self.param.reconMatrixF.objects, self.matrixF * self.recAcqRatioF)
            self.param.trigger('voxelF', 'reconVoxelF')
    
    
    @param.depends('matrixP', watch=True)
    def _watch_matrixP(self):
        with param.parameterized.batch_call_watchers(self):
            self.updateVoxelPobjects()
            self.voxelP = take_closest(self.param.voxelP.objects, self.FOVP/self.matrixP)
            add_to_pipeline(self.acquisitionPipeline, ['sampleKspace', 'updateSamplingTime', 'modulateKspace', 'simulateNoise', 'compileKspace'])
            add_to_pipeline(self.reconPipeline, ['partialFourierRecon', 'apodization', 'zerofill', 'reconstruct'])
            add_to_pipeline(self.sequencePipeline, ['setupPhasers', 'updateFOVPbounds', 'updateTurboFactorBounds', 'updateEPIfactorObjects'])
            self.param.reconMatrixP.objects = [m for m in reconMatrixValues if m>=self.matrixP]
            self.updateReconVoxelPobjects()
            self.reconMatrixP = take_closest(self.param.reconMatrixP.objects, self.matrixP * self.recAcqRatioP)
            self.param.trigger('voxelP', 'reconVoxelP')


    @param.depends('voxelF', watch=True)
    def _watch_voxelF(self):
        if self.param.voxelF.precedence > 0:
            self.matrixF = take_closest(self.param.matrixF.objects, self.FOVF/self.voxelF)
            add_to_pipeline(self.sequencePipeline, ['updateFOVFbounds'])


    @param.depends('voxelP', watch=True)
    def _watch_voxelP(self):
        if self.param.voxelP.precedence > 0:
            self.matrixP = take_closest(self.param.matrixP.objects, self.FOVP/self.voxelP)
            add_to_pipeline(self.sequencePipeline, ['updateFOVPbounds'])


    @param.depends('reconVoxelF', watch=True)
    def _watch_reconVoxelF(self):
        if self.param.reconVoxelF.precedence > 0:
            self.reconMatrixF = take_closest(self.param.reconMatrixF.objects, self.FOVF/self.reconVoxelF)
            add_to_pipeline(self.sequencePipeline, ['updateFOVFbounds'])


    @param.depends('reconVoxelP', watch=True)
    def _watch_reconVoxelP(self):
        if self.param.reconVoxelP.precedence > 0:
            self.reconMatrixP = take_closest(self.param.reconMatrixP.objects, self.FOVP/self.reconVoxelP)
            add_to_pipeline(self.sequencePipeline, ['updateFOVPbounds'])


    @param.depends('sliceThickness', watch=True)
    def _watch_sliceThickness(self):
        add_to_pipeline(self.acquisitionPipeline, ['sampleKspace', 'updateSamplingTime', 'modulateKspace', 'simulateNoise', 'compileKspace'])
        add_to_pipeline(self.reconPipeline, ['partialFourierRecon', 'apodization', 'zerofill', 'reconstruct'])
        add_to_pipeline(self.sequencePipeline, ['setupSliceSelection', 'placeFatSat'])
    
    
    @param.depends('frequencyDirection', watch=True)
    def _watch_frequencyDirection(self):
        add_to_pipeline(self.acquisitionPipeline, ['sampleKspace', 'updateSamplingTime', 'modulateKspace', 'simulateNoise', 'compileKspace'])
        add_to_pipeline(self.reconPipeline, ['partialFourierRecon', 'apodization', 'zerofill', 'reconstruct'])
        for p in [self.param.FOVF, self.param.FOVP, self.param.matrixF, self.param.matrixP, self.param.reconMatrixF, self.param.reconMatrixP]:
            if ' x' in p.label:
                p.label = p.label.replace(' x', ' y')
            elif ' y' in p.label:
                p.label = p.label.replace(' y', ' x')
        self.setup_frequency_encoding() # frequency oversampling is adapted to phantom FOV for efficiency
        self.setup_phase_encoding() # frequency oversampling is adapted to phantom FOV for efficiency
        add_to_pipeline(self.sequencePlotPipeline, ['calculate_k_trajectory'])
    

    @param.depends('trajectory', watch=True)
    def _watch_trajectory(self):
        add_to_pipeline(self.acquisitionPipeline, ['sampleKspace', 'updateSamplingTime', 'modulateKspace', 'simulateNoise', 'compileKspace'])
        add_to_pipeline(self.reconPipeline, ['partialFourierRecon', 'apodization', 'zerofill', 'reconstruct'])
        match(self.trajectory):
            case 'Cartesian':
                self.param.matrixF.label = 'Acquisition matrix {}'.format('y' if self.frequencyDirection=='anterior-posterior' else 'x')
                self.param.matrixP.label = 'Acquisition matrix {}'.format('x' if self.frequencyDirection=='anterior-posterior' else 'y')
                self.param.frequencyDirection.precedence = 1
                self.param.phaseOversampling.precedence = 3
            case 'Radial':
                self.param.matrixF.label = 'Acquisition matrix radial'
                self.param.matrixP.label = 'Number of spokes'
                self.param.frequencyDirection.precedence = -1
                self.param.phaseOversampling.precedence = -3
        self.setup_frequency_encoding()
        self.setup_phase_encoding()
        add_to_pipeline(self.sequencePlotPipeline, ['calculate_k_trajectory'])


    @param.depends('fieldStrength', watch=True)
    def _watch_fieldStrength(self):
        with param.parameterized.batch_call_watchers(self):
            self.updateFWshiftObjects()
            self.FWshift = take_closest(self.param.FWshift.objects, pixelBW2shift(self.pixelBandWidth, self.fieldStrength))
            self.param.trigger('FWshift')
            add_to_pipeline(self.acquisitionPipeline, ['updateSamplingTime', 'modulateKspace', 'simulateNoise', 'updatePDandT1w', 'compileKspace'])
            add_to_pipeline(self.reconPipeline, ['partialFourierRecon', 'apodization', 'zerofill', 'reconstruct'])
            self._watch_FatSat() # since fatsat pulse duration depends on fieldStrength
    

    @param.depends('pixelBandWidth', watch=True)
    def _watch_pixelBandWidth(self):
        with param.parameterized.batch_call_watchers(self):
            self.FWshift = take_closest(self.param.FWshift.objects, pixelBW2shift(self.pixelBandWidth, self.fieldStrength))
            self.FOVbandwidth = take_closest(self.param.FOVbandwidth.objects, pixelBW2FOVBW(self.pixelBandWidth, self.matrixF))
            add_to_pipeline(self.acquisitionPipeline, ['updateSamplingTime', 'modulateKspace', 'simulateNoise', 'compileKspace'])
            add_to_pipeline(self.reconPipeline, ['partialFourierRecon', 'apodization', 'zerofill', 'reconstruct'])
            add_to_pipeline(self.sequencePipeline, ['setupReadouts', 'updateMatrixFbounds', 'updateFOVFbounds', 'updateMatrixPbounds', 'updateFOVPbounds'])
    
    @param.depends('FWshift', watch=True)
    def _watch_FWshift(self):
        if self.param.FWshift.precedence > 0:
            self.pixelBandWidth = take_closest(self.param.pixelBandWidth.objects, shift2pixelBW(self.FWshift, self.fieldStrength))
    

    @param.depends('FOVbandwidth', watch=True)
    def _watch_FOVbandwidth(self):
        if self.param.FOVbandwidth.precedence > 0:
            self.pixelBandWidth = take_closest(self.param.pixelBandWidth.objects, FOVBW2pixelBW(self.FOVbandwidth, self.matrixF))


    @param.depends('NSA', watch=True)
    def _watch_NSA(self):
        add_to_pipeline(self.acquisitionPipeline, ['updateSamplingTime', 'modulateKspace', 'simulateNoise', 'compileKspace'])
        add_to_pipeline(self.reconPipeline, ['partialFourierRecon', 'apodization', 'zerofill', 'reconstruct'])


    @param.depends('partialFourier', watch=True)
    def _watch_partialFourier(self):
        add_to_pipeline(self.acquisitionPipeline, ['sampleKspace', 'updateSamplingTime', 'modulateKspace', 'simulateNoise', 'compileKspace'])
        add_to_pipeline(self.reconPipeline, ['partialFourierRecon', 'apodization', 'zerofill', 'reconstruct'])
        add_to_pipeline(self.sequencePipeline, ['setupRefocusing', 'setupReadouts', 'setupPhasers', 'updateMinTE', 'updateMinTR', 'updateMaxTE', 'updateMaxTI', 'updateBWbounds', 'updateMatrixFbounds', 'updateMatrixPbounds', 'updateFOVFbounds',  'updateFOVPbounds', 'updateSliceThicknessBounds', 'updateTurboFactorBounds', 'updateEPIfactorObjects'])


    @param.depends('turboFactor', watch=True)
    def _watch_turboFactor(self):
        add_to_pipeline(self.acquisitionPipeline, ['sampleKspace', 'updateSamplingTime', 'modulateKspace', 'simulateNoise', 'compileKspace'])
        add_to_pipeline(self.reconPipeline, ['partialFourierRecon', 'apodization', 'zerofill', 'reconstruct'])
        add_to_pipeline(self.sequencePipeline, ['setupRefocusing', 'setupReadouts', 'setupPhasers', 'updateMinTE', 'updateMinTR', 'updateMaxTE', 'updateMaxTI', 'updateBWbounds', 'updateMatrixFbounds', 'updateMatrixPbounds', 'updateFOVFbounds',  'updateFOVPbounds', 'updateSliceThicknessBounds', 'updateEPIfactorObjects'])
        self.updateEPIfactorObjects()


    @param.depends('EPIfactor', watch=True)
    def _watch_EPIfactor(self):
        add_to_pipeline(self.acquisitionPipeline, ['sampleKspace', 'updateSamplingTime', 'modulateKspace', 'simulateNoise', 'compileKspace'])
        add_to_pipeline(self.reconPipeline, ['partialFourierRecon', 'apodization', 'zerofill', 'reconstruct'])
        add_to_pipeline(self.sequencePipeline, ['setupReadouts', 'setupPhasers', 'updateMinTE', 'updateMinTR', 'updateMaxTE', 'updateMaxTI', 'updateBWbounds', 'updateMatrixFbounds', 'updateMatrixPbounds', 'updateFOVFbounds',  'updateFOVPbounds', 'updateSliceThicknessBounds', 'updateTurboFactorBounds'])
        self.updateTurboFactorBounds()
    

    @param.depends('shot', watch=True)
    def _watch_shot(self):
        add_to_pipeline(self.sequencePipeline, ['setupPhasers'])
        add_to_pipeline(self.sequencePlotPipeline, ['renderSignalBoard'])
    

    @param.depends('sequence', watch=True)
    def _watch_sequence(self):
        add_to_pipeline(self.acquisitionPipeline, ['modulateKspace', 'updatePDandT1w', 'compileKspace'])
        add_to_pipeline(self.reconPipeline, ['partialFourierRecon', 'apodization', 'zerofill', 'reconstruct'])
        add_to_pipeline(self.sequencePipeline, ['setupExcitation', 'setupRefocusing', 'setupInversion', 'setupPhasers', 'placeReadouts', 'placePhasers'])
        self.param.FA.precedence = 1 if self.sequence=='Spoiled Gradient Echo' else -1
        self.param.TI.precedence = 1 if self.sequence=='Inversion Recovery' else -1
        if self.sequence=='Spoiled Gradient Echo':
            self.turboFactor = 1
            self.param.turboFactor.precedence = -6
        else:
            self.param.turboFactor.precedence = 6
    

    @param.depends('TE', watch=True)
    def _watch_TE(self):
        add_to_pipeline(self.acquisitionPipeline, ['modulateKspace', 'updatePDandT1w', 'compileKspace'])
        add_to_pipeline(self.reconPipeline, ['partialFourierRecon', 'apodization', 'zerofill', 'reconstruct'])
        add_to_pipeline(self.sequencePipeline, ['setupPhasers', 'placeRefocusing', 'placeReadouts', 'updateMatrixFbounds', 'updateFOVFbounds', 'updateMatrixPbounds', 'updateFOVPbounds'])
    

    @param.depends('TR', watch=True)
    def _watch_TR(self):
        add_to_pipeline(self.acquisitionPipeline, ['updatePDandT1w', 'compileKspace'])
        add_to_pipeline(self.reconPipeline, ['partialFourierRecon', 'apodization', 'zerofill', 'reconstruct'])
        add_to_pipeline(self.sequencePipeline, ['updateMaxTE', 'updateMaxTI', 'updateBWbounds', 'updateSliceThicknessBounds'])
        add_to_pipeline(self.sequencePlotPipeline, ['renderTRspan'])
    

    @param.depends('TI', watch=True)
    def _watch_TI(self):
        add_to_pipeline(self.acquisitionPipeline, ['updatePDandT1w', 'compileKspace'])
        add_to_pipeline(self.reconPipeline, ['partialFourierRecon', 'apodization', 'zerofill', 'reconstruct'])
        add_to_pipeline(self.sequencePipeline, ['placeInversion'])


    @param.depends('FA', watch=True)
    def _watch_FA(self):
        add_to_pipeline(self.acquisitionPipeline, ['updatePDandT1w', 'compileKspace'])
        add_to_pipeline(self.reconPipeline, ['partialFourierRecon', 'apodization', 'zerofill', 'reconstruct'])
        add_to_pipeline(self.sequencePipeline, ['setupExcitation'])
    
    
    @param.depends('FatSat', watch=True)
    def _watch_FatSat(self):
        add_to_pipeline(self.acquisitionPipeline, ['compileKspace'])
        add_to_pipeline(self.reconPipeline, ['partialFourierRecon', 'apodization', 'zerofill', 'reconstruct'])
        add_to_pipeline(self.sequencePipeline, ['setupFatSat', 'updateMaxTE', 'updateBWbounds'])
    

    @param.depends('homodyne', watch=True)
    def _watch_homodyne(self):
        add_to_pipeline(self.reconPipeline, ['partialFourierRecon', 'apodization', 'zerofill', 'reconstruct'])

    
    @param.depends('doApodize', watch=True)
    def _watch_apodiztion(self):
        if self.doApodize:
            self.param.apodizationAlpha.precedence = abs(self.param.apodizationAlpha.precedence)
        else:
            self.param.apodizationAlpha.precedence = -abs(self.param.apodizationAlpha.precedence)
        add_to_pipeline(self.reconPipeline, ['apodization', 'zerofill', 'reconstruct'])
        
    
    @param.depends('apodizationAlpha', watch=True)
    def _watch_apodiztionAlpha(self):
        add_to_pipeline(self.reconPipeline, ['apodization', 'zerofill', 'reconstruct'])

    
    @param.depends('doZerofill', watch=True)
    def _watch_zerofill(self):
        add_to_pipeline(self.reconPipeline, ['zerofill', 'reconstruct'])


    @param.depends('reconMatrixF', watch=True)
    def _watch_reconMatrixF(self):
        self.recAcqRatioF = self.reconMatrixF / self.matrixF
        if self.doZerofill:
            add_to_pipeline(self.reconPipeline, ['zerofill', 'reconstruct'])
        self.reconVoxelF = take_closest(self.param.reconVoxelF.objects, self.FOVF/self.reconMatrixF)


    @param.depends('reconMatrixP', watch=True)
    def _watch_reconMatrixP(self):
        self.recAcqRatioP = self.reconMatrixP / self.matrixP
        if self.doZerofill:
            add_to_pipeline(self.reconPipeline, ['zerofill', 'reconstruct'])
        self.reconVoxelP = take_closest(self.param.reconVoxelP.objects, self.FOVP/self.reconMatrixP)
    

    def getSeqStart(self):
        if self.sequence == 'Inversion Recovery': 
            return self.boards['slice']['objects']['slice select inversion']['time'][0]
        elif self.FatSat:
            return self.boards['RF']['objects']['fatsat']['time'][0]
        else:
            return self.boards['slice']['objects']['slice select excitation']['time'][0]
    
    
    def get_TE_from_centermost_echoes(self, readtrain_spacing, centermost_gr_echoes, centermost_rf_echoes):
        TE = readtrain_spacing * (1 + np.mean(centermost_rf_echoes))
        readtrain_shift = self.gr_echo_spacing * (np.mean(centermost_gr_echoes) - (self.EPIfactor-1)/2)
        TE += readtrain_shift
        return TE
    
    
    def updateMinTE(self):
        min_readtrain_spacing = self.get_min_readtrain_spacing()
        if self.EPIfactor == 1: # flexible segment order for (turbo) spin echo
            min_centermost_gr_echoes = [0]
            min_centermost_rf_echoes = [0]
            if (self.split_center and self.turboFactor > 1):
                min_centermost_rf_echoes += [1]
            self.minTE = self.get_TE_from_centermost_echoes(min_readtrain_spacing, min_centermost_gr_echoes, min_centermost_rf_echoes)
        else: # linear segment order for EPI and GRASE
            # # pick forward or reverse order that minimizes TE (may be forward for GRASE)
            TEcands = [self.get_TE_from_centermost_echoes(min_readtrain_spacing, *self.get_centermost_echoes_linear_order(reverse)) for reverse in [True, False]]
            self.minTE = min(TEcands)
        add_to_pipeline(self.sequencePipeline, ['updateMaxTE'])
    
    
    def updateMinTR(self):
        self.minTR = self.boards['slice']['objects']['spoiler']['time'][-1]
        self.minTR -= self.getSeqStart()
        self.setParamDiscreteBounds(self.param.TR, TRvalues, minval=self.minTR)
        add_to_pipeline(self.sequencePipeline, ['updateMaxTE', 'updateMaxTI'])
    

    def updateMaxTE(self):
        maxTE = self.TR - self.minTR + self.TE
        self.setParamDiscreteBounds(self.param.TE, TEvalues, minval=self.minTE, maxval=maxTE)
    
    
    def updateMaxTI(self):
        if self.sequence != 'Inversion Recovery':
            return
        maxTI = self.TR - self.minTR + self.TI
        self.setParamDiscreteBounds(self.param.TI, TIvalues, minval=40, maxval=maxTI)
    

    def resolveConflicts(self):
        if not self.outbound_params:
            return
        if 'TR' in self.outbound_params:
            print('Warning: Resolving conflict: TR')
            self.outbound_params.remove('TR')
            tr = self.TR
            self.param.TR.objects = TRvalues
            add_to_pipeline(self.sequencePipeline, ['updateMinTR'])
            self.TR = TRvalues[-1] # max TR
            self.TR = take_closest(self.param.TR.objects, tr) # Set back TR within (new) bounds
        if 'TI' in self.outbound_params:
            print('Warning: Resolving conflict: TI')
            self.outbound_params.remove('TI')
            self.TI = take_closest(self.param.TI.objects, self.TI) # Set back TI within (new) bounds
        if 'TE' in self.outbound_params:
            print('Warning: Resolving conflict: TE')
            self.outbound_params.remove('TE')
            self.param.TE.objects = TEvalues
            self.TE = take_closest([t for t in TEvalues if t>=self.minTE], self.TE)
        elif self.outbound_params:
            print('Warning: Unresolved conflict:', self.outbound_params)
    

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
        maxReadoutAreas = []
        # See paramBounds.tex for formulae
        d = 1e3 / self.pixelBandWidth # readout duration
        s = self.maxSlew
        if isGradientEcho(self.sequence):
            centermost_gr_echoes, centermost_rf_echoes = self.get_centermost_echoes_linear_order(reverse=True)
            if len(centermost_gr_echoes)==1:
                N = centermost_gr_echoes[0] + 1/2
                M = centermost_gr_echoes[0] * 2 + 1
            else:
                N = max(centermost_gr_echoes)
                M = N * 2
            t = self.TE - self.boards['RF']['objects']['excitation']['dur_f']/2
            v = 0 # gap between readouts
            for _ in range(2): # update readout gap after first pass
                if (M > 1):
                    # max wrt G slice or G phase:
                    q = t - max(self.maxPhaserDuration,
                                self.boards['slice']['objects']['slice select excitation']['riseTime_f'] + self.boards['slice']['objects']['slice select rephaser']['dur_f'])
                    A = d*s*(q - N*(d+v) + v/2) / (M-1) # eq. 15
                    maxReadoutAreas.append(A)
                # max wrt G read:
                h_roots = np.roots([8*(3-2*M), 4*s*(t*(2*M-6)+d*(2*N-M)+v*(2*N-1)), s**2*(4*t**2-d**2)]) # eq. 12
                h = min([h for h in h_roots if h>0] + [self.maxAmp]) # truncate prephaser amp to max amp
                A_roots = np.roots([1, d*(d*s + 2*M*h), d**2*h*(2*h-s*(2*t-2*N*(d+v)+v))]) # eq. 13
                if np.all(A_roots<0):
                    return 0 # no positive roots
                A = min([A for A in A_roots if A>0])
                maxReadoutAreas.append(A)
                read_risetime = min(maxReadoutAreas) / (d * s)
                v = max(self.max_blip_dur - 2 * read_risetime, 0)
        else: # (turbo) spin echo / GRASE
            # limit by half readout duration tr:
            tr = (self.readtrain_spacing - self.boards['RF']['objects']['refocusing'][0]['dur_f']) / self.EPIfactor / 2
            Ar = d*s* tr - d**2*s/2
            maxReadoutAreas.append(Ar)
            # limit by prephaser duration tp:
            tp = (self.readtrain_spacing - self.boards['RF']['objects']['refocusing'][0]['dur_f'] - self.boards['RF']['objects']['excitation']['dur_f'])/2
            h = s * tp / 2
            h = min(h, self.maxAmp)
            Ap = d * (np.sqrt((d*s)**2 - 8*h*(h-s*tp)) - d*s) / 2
            maxReadoutAreas.append(Ap)
        maxReadoutAreas.append(self.maxAmp * 1e3 / self.pixelBandWidth) # max wrt maxAmp
        return min(maxReadoutAreas)
    

    def getMaxPhaserArea(self):
        if isGradientEcho(self.sequence):
            maxPhaserDuration = self.readtrain_spacing - self.boards['RF']['objects']['excitation']['dur_f']/2 - self.gre_echo_train_dur/2 + self.readout_risetime
        else:
            maxPhaserDuration = (self.readtrain_spacing - self.boards['RF']['objects']['refocusing'][0]['dur_f'] - self.gre_echo_train_dur)/2 + self.readout_risetime
        maxRiseTime = self.maxAmp / self.maxSlew
        if maxPhaserDuration > 2 * maxRiseTime: # trapezoid maxPhaser
            maxPhaserArea = (maxPhaserDuration - maxRiseTime) * self.maxAmp
        else: # triangular maxPhaser
            maxPhaserArea = (maxPhaserDuration/2)**2 * self.maxSlew
        return maxPhaserArea
    

    def get_max_read_duration(self, read_start_to_kcenter, kcenter_to_read_end, reverse=False):
        # simplification: use current risetime (self.readout_risetime)
        centermost_gr_echoes, centermost_rf_echoes = self.get_centermost_echoes_linear_order(reverse=reverse)
        if len(centermost_gr_echoes)==1:
            nEarlyReadouts = centermost_gr_echoes[0] + 1/2
            nEarlyRamps = centermost_gr_echoes[0] * 2
        else:
            nEarlyReadouts = max(centermost_gr_echoes)
            nEarlyRamps = nEarlyReadouts * 2 - 1
        # max limit imposed by TE:
        maxReadDurEarly = ((read_start_to_kcenter - nEarlyRamps * self.readout_risetime) / nEarlyReadouts)
        nLateReadouts = self.EPIfactor - nEarlyReadouts
        nLateRamps = (self.EPIfactor - 1) * 2 - nEarlyRamps
        # max limit imposed by TR:
        maxReadDurLate = ((kcenter_to_read_end - nLateRamps * self.readout_risetime) / nLateReadouts)
        return min(maxReadDurEarly, maxReadDurLate)
    

    def updateBWbounds(self):
        # See paramBounds.tex for formulae relating to the readout board
        s = self.maxSlew
        A = 1e3 * self.matrixF / (self.FOVF * constants.GYRO) # readout area
        minReadDurations = [.5] # msec (corresponds to a pixel BW of 2000 Hz)
        # min limit imposed by maximum gradient amplitude:
        minReadDurations.append(A / self.maxAmp)
        maxReadDurations = [8.] # msec (corresponds to a pixel BW of 125 Hz)
        if isGradientEcho(self.sequence):
            minPhaserTime = min([self.boards['phase']['objects'][typ][0]['dur_f'] for typ in ['phasers', 'rephasers']])
            readStartToTE = self.TE - self.boards['RF']['objects']['excitation']['dur_f']/2
            readStartToTE -= max(
                self.boards['frequency']['objects']['read prephaser']['dur_f'] + self.readout_risetime, # TODO: consider maximum dur+risetime, not only current (difficult!)
                minPhaserTime,
                self.boards['slice']['objects']['slice select excitation']['riseTime_f'] + self.boards['slice']['objects']['slice select rephaser']['dur_f'])
            TEtoSpoiler = (self.TR - (-self.getSeqStart()) - self.boards['slice']['objects']['spoiler']['dur_f']) - self.TE
            # pick forward or reverse order that maximizes read duration limit
            maxReadDurs = []
            for reverse in [True, False]:
                maxReadDurs.append(self.get_max_read_duration(readStartToTE, TEtoSpoiler, reverse))
            maxReadDurations.append(max(maxReadDurs))
        else: # spin echo
            refocusing_dur = self.boards['RF']['objects']['refocusing'][0]['dur_f']
            if self.turboFactor==1 and self.EPIfactor==1: # prephaser should only be limiting for pure spin echo
                # min limit imposed by prephaser duration tp:
                tp = self.TE/2 - refocusing_dur/2 - self.boards['RF']['objects']['excitation']['dur_f']/2
                h = s * tp / 2
                h = min(h, self.maxAmp)
                minReadDurations.append(np.sqrt(A**2/(2*h*s*tp - s*A - 2*h**2)))
            if self.EPIfactor==1:
                max_readtrain_spacing = self.TE / (1 + 1/2 * self.split_center)
            else: # linear k-space order
                # TODO: correct this
                max_readtrain_spacing = max([self.get_readtrain_spacing_linear_order(reverse) for reverse in [True, False]])
            idle_space = max_readtrain_spacing - self.boards['RF']['objects']['refocusing'][0]['dur_f']
            # max limit imposed by phaser:
            maxReadDurations.append((idle_space - 2 * self.maxPhaserDuration - self.max_blip_dur * (self.EPIfactor-1))/self.EPIfactor)
            # tr is half the maximum readout gradient duration
            tr = ((idle_space) / self.EPIfactor) / 2
            # max limit imposed by readout rise time:
            maxReadDurations.append(tr + np.sqrt(tr**2 - 2*A/s))
            # max limit imposed by slice select refocusing down ramp time:
            maxReadDurations.append((tr - self.boards['slice']['objects']['slice select refocusing'][0]['riseTime_f']) * 2)
            # readtrain_spacing may be limited by TR:
            read_end_by_TR = (self.TR - (-self.getSeqStart()) - self.boards['slice']['objects']['spoiler']['dur_f'])
            read_end_by_else = self.readtrain_spacing * (self.turboFactor + 1/2) - refocusing_dur/2
            maxReadDurations.append((tr - (read_end_by_else-read_end_by_TR)) * 2)
        small = 1e-2 # to avoid roundoff errors
        minpBW = 1e3 / min(maxReadDurations) + small
        maxpBW = 1e3 / max(minReadDurations) - small
        self.setParamDiscreteBounds(self.param.pixelBandWidth, pBWvalues, minval=minpBW, maxval=maxpBW)
        self.updateFWshiftObjects()
        self.updateFOVbandwidthObjects()
        
        
    def updateMatrixFbounds(self):
        minMatrixF, maxMatrixF = 16, 600
        maxMatrixF = min(maxMatrixF, self.getMaxReadoutArea() * 1e-3 * self.FOVF * constants.GYRO)
        if self.parameterStyle == 'Matrix and FOV BW':
            minMatrixF = max(minMatrixF, self.pixelBandWidth * self.matrixF / self.param.pixelBandWidth.objects[-1])
            maxMatrixF = min(maxMatrixF, self.pixelBandWidth * self.matrixF / self.param.pixelBandWidth.objects[0])
        self.setParamDiscreteBounds(self.param.matrixF, matrixValues, minval=minMatrixF, maxval=maxMatrixF)
        self.updateVoxelFobjects()
        self.updateReconVoxelFobjects()
    

    def updateMatrixPbounds(self):
        maxMatrixP = int(self.getMaxPhaserArea() * 2e-3 * self.FOVP * constants.GYRO) + 1
        self.setParamDiscreteBounds(self.param.matrixP, matrixValues, maxval=maxMatrixP)
        self.updateVoxelPobjects()
        self.updateReconVoxelPobjects()


    def updateFOVFbounds(self):
        maxReadoutArea = self.getMaxReadoutArea()
        minFOV = 1e3 * self.matrixF / (maxReadoutArea * constants.GYRO) if maxReadoutArea > 0 else 600
        minFOV = max(minFOV, 100)
        maxFOV = 600
        if self.parameterStyle == 'Voxelsize and Fat/water shift':
            minFOV = max(minFOV, self.FOVF/self.matrixF * self.param.matrixF.objects[0])
            minFOV = max(minFOV, self.FOVF/self.reconMatrixF * self.param.reconMatrixF.objects[0])
            maxFOV = min(maxFOV, self.FOVF/self.matrixF * self.param.matrixF.objects[-1])
            maxFOV = min(maxFOV, self.FOVF/self.reconMatrixF * self.param.reconMatrixF.objects[-1])
        self.setParamBounds(self.param.FOVF, minFOV, maxFOV)


    def updateFOVPbounds(self):
        minFOV = (self.matrixP - 1) / (self.getMaxPhaserArea() * constants.GYRO * 2e-3)
        minFOV = max(minFOV, 100)
        maxFOV = 600
        if self.parameterStyle == 'Voxelsize and Fat/water shift':
            minFOV = max(minFOV, self.FOVP/self.matrixP * self.param.matrixP.objects[0])
            minFOV = max(minFOV, self.FOVP/self.reconMatrixP * self.param.reconMatrixP.objects[0])
            maxFOV = min(maxFOV, self.FOVP/self.matrixP * self.param.matrixP.objects[-1])
            maxFOV = min(maxFOV, self.FOVP/self.reconMatrixP * self.param.reconMatrixP.objects[-1])
        self.setParamBounds(self.param.FOVP, minFOV, maxFOV)


    def updateFWshiftObjects(self):
        self.param.FWshift.objects = unique_list([float('{:.2f}'.format(pixelBW2shift(pBW, self.fieldStrength))) for pBW in self.param.pixelBandWidth.objects[::-1]])
    

    def updateFOVbandwidthObjects(self):
        self.param.FOVbandwidth.objects = unique_list([float('{:.2g}'.format(pixelBW2FOVBW(pBW, self.matrixF))) for pBW in self.param.pixelBandWidth.objects])
    
    
    def updateVoxelFobjects(self):
        self.param.voxelF.objects = unique_list([float('{:.3g}'.format(self.FOVF/matrix)) for matrix in self.param.matrixF.objects[::-1]])


    def updateVoxelPobjects(self):
        self.param.voxelP.objects = unique_list([float('{:.3g}'.format(self.FOVP/matrix)) for matrix in self.param.matrixP.objects[::-1]])


    def updateReconVoxelFobjects(self):
        self.param.reconVoxelF.objects = unique_list([float('{:.3g}'.format(self.FOVF/matrix)) for matrix in self.param.reconMatrixF.objects[::-1]])


    def updateReconVoxelPobjects(self):
        self.param.reconVoxelP.objects = unique_list([float('{:.3g}'.format(self.FOVP/matrix)) for matrix in self.param.reconMatrixP.objects[::-1]])


    def updateSliceThicknessBounds(self):
        minThks = [.5]
        minThks.append(self.boards['RF']['objects']['excitation']['FWHM_f'] / (self.maxAmp * constants.GYRO))
        if not isGradientEcho(self.sequence):
            minThks.append(self.boards['RF']['objects']['refocusing'][0]['FWHM_f'] / (self.maxAmp * constants.GYRO))
        if self.sequence=='Inversion Recovery':
            minThks.append(self.boards['RF']['objects']['inversion']['FWHM_f'] / (self.maxAmp * constants.GYRO) * self.inversionThkFactor)
        
        # Constraint due to TR: 
        if self.sequence=='Inversion Recovery':
            maxRiseTime = self.TR - (self.boards['slice']['objects']['spoiler']['time'][-1] - self.boards['RF']['objects']['inversion']['time'][0])
            maxAmp = self.maxSlew * maxRiseTime
            minThks.append(self.boards['RF']['objects']['inversion']['FWHM_f'] / (maxAmp * constants.GYRO))
        else:
            maxRiseTime = self.TR - (self.boards['slice']['objects']['spoiler']['time'][-1] - self.boards['RF']['objects']['excitation']['time'][0])
            maxAmp = self.maxSlew * maxRiseTime
            minThks.append(self.boards['RF']['objects']['excitation']['FWHM_f'] / (maxAmp * constants.GYRO))
        
        # See paramBounds.tex for formulae
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
        minThks.append(Be * d / (constants.GYRO * A)) # mm
        
        self.setParamBounds(self.param.sliceThickness, max(minThks), 10.)


    def updateTurboFactorBounds(self):
        maxTurboFactor = int(np.floor(self.matrix[self.phaseDir] * self.partialFourier / self.EPIfactor * 2)) # let's limit phase oversampling to 2
        maxTurboFactor = min(maxTurboFactor, 64)

        # turboFactor must equal 1 when the EPIfactor is even
        if not self.EPIfactor%2:
            self.param.turboFactor.bounds = (1, 1)
            self.param.turboFactor.constant = True
        else:
            self.param.turboFactor.bounds = (1, maxTurboFactor)
            self.param.turboFactor.constant = False


    def updateEPIfactorObjects(self):
        maxEPIFactor = int(np.floor(self.matrix[self.phaseDir] * self.partialFourier / self.turboFactor * 2)) # let's limit phase oversampling to 2
        values = [v for v in EPIfactorValues if v <= maxEPIFactor]
        # EPIfactor must be odd for turbo spin echo (GRASE)
        if self.turboFactor > 1:
            self.param.EPIfactor.objects = [v for v in values if v%2]
        else:
            self.param.EPIfactor.objects = values


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
            print('Preparing k-space for "{}" phantom. This might take a few minutes on first use...'.format(self.object))
            polys = readSVG(Path(phantomPath / self.object).with_suffix('.svg'))
            self.tissues = set([poly['tissue'] for poly in polys])
            self.phantom['kspace'] = {tissue: np.zeros((self.phantom['matrix']), dtype=complex) for tissue in self.tissues}
            k = np.array(np.meshgrid(self.phantom['kAxes'][0], self.phantom['kAxes'][1])).T
            for poly in tqdm(polys):
                self.phantom['kspace'][poly['tissue']] += kspacePolygon(poly, k)
            for tissue in self.tissues:
                file = Path(phantomPath / tissue).with_suffix('.npy')
                np.save(file, self.phantom['kspace'][tissue])
            print('DONE')
        self.setup_frequency_encoding() # frequency oversampling is adapted to phantom FOV for efficiency


    def get_min_readtrain_spacing(self):
        # Get shortest spacing for (center of) gradient echo trains
        # Equals center position of gradient echo (train) for gradient echo sequences
        # Equals rf echo spacing for spin echo sequences
        max_phaser_area = np.min(self.kAxes[self.phaseDir]) * 1e3 / constants.GYRO   # uTs/m
        max_phaser_dur = sequence.getGradient('_', totalArea=max_phaser_area)['dur_f']
        if isGradientEcho(self.sequence):
            spacing = (self.boards['RF']['objects']['excitation']['dur_f'] + self.gre_echo_train_dur) / 2 - self.readout_risetime
            spacing += max(
                self.boards['frequency']['objects']['read prephaser']['dur_f'] + self.readout_risetime,
                max_phaser_dur,
                self.boards['slice']['objects']['slice select excitation']['riseTime_f'] + self.boards['slice']['objects']['slice select rephaser']['dur_f']
            )
        else: # spin echo
            # before refocusing pulse:
            leftSide = (self.boards['RF']['objects']['excitation']['dur_f'] + self.boards['RF']['objects']['refocusing'][0]['dur_f']) / 2
            leftSide += max(
                self.boards['frequency']['objects']['read prephaser']['dur_f'], 
                self.boards['slice']['objects']['slice select excitation']['riseTime_f'] + self.boards['slice']['objects']['slice select rephaser']['dur_f'] + (self.boards['slice']['objects']['slice select refocusing'][0]['riseTime_f'])
            )
            # after refocusing pulse:
            rightSide = (self.boards['RF']['objects']['refocusing'][0]['dur_f'] + self.gre_echo_train_dur) / 2 - self.readout_risetime
            rightSide += max(
                self.readout_risetime,
                max_phaser_dur,
                self.boards['slice']['objects']['slice select refocusing'][0]['riseTime_f']
            )
            spacing = max(leftSide, rightSide) * 2
        return spacing
    
    
    def get_centermost_echoes_linear_order(self, reverse=False):
        # get index lists of rf echo(es) and gradient echo(es) closest to k-space center for linear k-space ordering
        centermost_gr_echoes = []
        centermost_rf_echoes = []
        central_segments = self.central_segments
        if reverse:
            central_segments = [self.num_segm - 1 - segm for segm in central_segments]
        for segm in central_segments:
            centermost_gr_echoes.append(segm // self.turboFactor)
            centermost_rf_echoes.append(segm % self.turboFactor)
        return centermost_gr_echoes, centermost_rf_echoes
    
    
    def get_readtrain_spacing_linear_order(self, reverse):
        centermost_gr_echoes, centermost_rf_echoes = self.get_centermost_echoes_linear_order(reverse)
        readtrain_shift = self.gr_echo_spacing * (np.mean(centermost_gr_echoes) - (self.EPIfactor-1)/2)
        central_rf_echo_time = self.TE - readtrain_shift
        readtrain_spacing = central_rf_echo_time / (1 + np.mean(centermost_rf_echoes))
        return readtrain_spacing
    
    
    def set_readtrain_spacing(self):
        # Equals center position of gradient echo (train) for gradient echo sequences
        # Equals rf echo spacing for spin echo sequences
        min_readtrain_spacing = self.get_min_readtrain_spacing()
        if self.EPIfactor == 1: # (turbo) spin echo
            minTE = min_readtrain_spacing * (1 + .5 * self.split_center)
            te = max(self.TE, minTE)
            self.centermost_rf_echo = int(np.floor(te / min_readtrain_spacing - (1 + .5 * self.split_center)))
            self.centermost_rf_echo = min(self.centermost_rf_echo, self.turboFactor - 1 - self.split_center)
            self.readtrain_spacing = te / (self.centermost_rf_echo + (1 + .5 * self.split_center))
        else: # linear k-space order for EPI / GRASE
            # pick forward or reverse order that minimizes spacing while respecting minimum spacing
            # TODO: respect TR as well
            cands = sorted([(self.get_readtrain_spacing_linear_order(reverse), reverse) for reverse in [True, False]])
            (self.readtrain_spacing, self.reverse_linear_order) = cands[0] if cands[0][0]>min_readtrain_spacing and cands[1][0]>min_readtrain_spacing else cands[1]
        add_to_pipeline(self.sequencePipeline, ['placeRefocusing', 'placeReadouts', 'placePhasers'])
    
    
    def setup_phase_encoding_table(self):
        if self.EPIfactor == 1: # (turbo) spin echo
            segment_order = get_segment_order(self.turboFactor, self.num_sym_segm, self.centermost_rf_echo)
            self.pe_table = [[[segment * self.num_shots + shot] for segment in segment_order] for shot in range(self.num_shots)]
        else: # EPI and GRASE
            order = -1 if self.reverse_linear_order else 1
            self.pe_table = [[list(range(rf_echo * self.num_shots + shot, self.num_measured_lines, self.num_shots * self.turboFactor))[::order] for rf_echo in range(self.turboFactor)][::order] for shot in range(self.num_shots)]
    

    def setup_frequency_encoding(self):
        self.freqDir = DIRECTIONS[self.frequencyDirection]
        self.FOV[self.freqDir] = self.FOVF
        self.matrix[self.freqDir] = self.matrixF
        self.oversampledMatrix[self.freqDir] = self.matrixF
        # at least Nyquist sampling wrt phantom if loaded
        if hasattr(self, 'phantom') and self.FOV[self.freqDir] < self.phantom['FOV'][self.freqDir]:
            self.oversampledMatrix[self.freqDir] = int(np.ceil(self.phantom['FOV'][self.freqDir] * self.matrix[self.freqDir] / self.FOV[self.freqDir]))
        self.kAxes[self.freqDir] = getKaxis(self.oversampledMatrix[self.freqDir], self.FOV[self.freqDir]/self.matrix[self.freqDir])
    

    def setup_phase_encoding(self):
        self.phaseDir = 1 - DIRECTIONS[self.frequencyDirection]
        self.FOV[self.phaseDir] = self.FOVP
        self.matrix[self.phaseDir] = self.matrixP
        
        self.num_shots = int(np.ceil(self.matrix[self.phaseDir] * (1 + self.phaseOversampling / 100) * self.partialFourier / self.turboFactor / self.EPIfactor))
        self.num_measured_lines = self.num_shots * self.turboFactor * self.EPIfactor
        self.oversampledPartialMatrix = self.num_measured_lines # Needs to be modified for parallel imaging
        # oversampling may be higher than prescribed since num_shots must be integer:
        self.oversampledMatrix[self.phaseDir] = max(self.oversampledPartialMatrix, int(np.ceil(self.matrix[self.phaseDir] * (1 + self.phaseOversampling / 100))))
        self.kAxes = [getKaxis(self.oversampledMatrix[dim], self.FOV[dim]/self.matrix[dim]) for dim in range(len(self.matrix))]
        # undersample by partial Fourier:
        self.kAxes[self.phaseDir] = self.kAxes[self.phaseDir][:self.oversampledPartialMatrix]
        assert(len(self.kAxes[self.phaseDir]) == self.num_measured_lines)

        self.num_segm = int(self.num_measured_lines / self.num_shots)
        num_sym_lines = 2 * self.num_measured_lines - self.oversampledMatrix[self.phaseDir]
        # check if center of k-space lies between two segments:
        self.split_center = (num_sym_lines % self.num_shots == 0) and ((num_sym_lines / self.num_shots) % 2 == 0)
        # number of k-space segments symmetric about center:
        if self.split_center:
            self.num_sym_segm = int(num_sym_lines / self.num_shots)
            self.central_segments = [self.num_segm - self.num_sym_segm//2 - 1, self.num_segm - self.num_sym_segm//2]
        else:
            self.num_sym_segm = int(np.round((num_sym_lines / self.num_shots - 1) / 2)) * 2 + 1
            self.central_segments = [self.num_segm - self.num_sym_segm//2 - 1]
        self.param.shot.bounds=(1, self.num_shots)
        self.shot = min(self.shot, self.num_shots)
    

    def sampleKspace(self):
        if self.trajectory=='Cartesian':
            self.plainKspaceComps = resampleKspaceCartesian(self.phantom, self.kAxes)
        else:
            kSamples = np.array(np.meshgrid(self.kAxes[0], self.kAxes[1])).T # TODO: properly
            self.plainKspaceComps = resampleKspace(self.phantom, kSamples)
        
        # Lorenzian line shape to mimic slice thickness
        blurFactor = .5
        sliceThicknessFilter = self.sliceThickness * np.outer(*[np.exp(-blurFactor * self.sliceThickness * np.abs(ax)) for ax in self.kAxes])
        for tissue in self.plainKspaceComps:
            self.plainKspaceComps[tissue] *= sliceThicknessFilter
        
        # signal for SNR calculation
        self.signal = np.sqrt(self.oversampledMatrix[self.freqDir] * self.num_measured_lines) * self.sliceThickness * np.prod(self.FOV)/np.prod(self.matrix)


    def updateSamplingTime(self):
        # time of sample along (positive) readout relative to (k-space) center
        self.samplingTime = self.kAxes[self.freqDir] * self.FOV[self.freqDir] / self.matrix[self.freqDir] / self.pixelBandWidth * 1e3 # msec
        self.noiseStd = self.noiseGain / np.sqrt(np.diff(self.samplingTime[:2]) * self.NSA) / self.fieldStrength
    

    def modulateKspace(self):
        pe_table = np.array(self.pe_table)
        TEs = np.zeros((pe_table.size))
        spin_echoes = np.zeros((pe_table.size))
        reverse = np.zeros((pe_table.size), dtype=bool)
        for ky in range(pe_table.size):
            shot, rf_echo, gr_echo = np.argwhere(pe_table==ky)[0]
            TEs[ky] = self.boards['frequency']['objects']['readouts'][rf_echo][gr_echo]['center_f']
            spin_echoes[ky] = (rf_echo + 1) * self.readtrain_spacing
            reverse[ky] = self.boards['frequency']['objects']['readouts'][rf_echo][gr_echo]['area_f'] < 0
        samplingTime = np.expand_dims(self.samplingTime, axis=[dim for dim in range(len(self.matrix)) if dim != self.freqDir])
        self.timeAfterExcitation = samplingTime + np.expand_dims(TEs, axis=[dim for dim in range(len(self.matrix)) if dim != self.phaseDir])
        # EPI rowflip:
        reverseTimeAfterExcitation = np.flip(self.timeAfterExcitation, axis=self.freqDir)
        reverse = np.expand_dims(reverse, axis=[dim for dim in range(len(self.matrix)) if dim != self.phaseDir])
        reverse = reverse.repeat(len(self.samplingTime), axis=self.freqDir)
        self.timeAfterExcitation[reverse] = reverseTimeAfterExcitation[reverse]
        
        timeRelativeInphase = self.timeAfterExcitation.copy()
        if not isGradientEcho(self.sequence):
            spin_echoes = np.expand_dims(spin_echoes, axis=[dim for dim in range(len(self.matrix)) if dim != self.phaseDir])
            timeRelativeInphase -= spin_echoes # for spinecho, subtract Hahn echo position from timeAfterExcitation
        
        self.kspaceComps = {}
        for tissue in self.tissues:
            T2w = getT2w(tissue, self.timeAfterExcitation, timeRelativeInphase, self.fieldStrength)
            if TISSUES[tissue]['FF'] == .00:
                self.kspaceComps[tissue] = self.plainKspaceComps[tissue] * T2w
            else: # fat containing tissues
                self.kspaceComps[tissue + 'Water'] = self.plainKspaceComps[tissue] * T2w
                for component, resonance in FATRESONANCES.items():
                    T2w = getT2w(component, self.timeAfterExcitation, timeRelativeInphase, self.fieldStrength)
                    dephasing = np.exp(2j*np.pi * constants.GYRO * self.fieldStrength * resonance['shift'] * timeRelativeInphase * 1e-3)
                    self.kspaceComps[tissue + component] = self.plainKspaceComps[tissue] * dephasing * T2w
            if tissue==self.phantom['referenceTissue']:
                self.decayedSignal = self.signal * np.take(np.take(T2w, np.argmin(np.abs(self.kAxes[self.freqDir])), axis=self.freqDir), np.argmin(np.abs(self.kAxes[self.phaseDir])))
    
    
    def simulateNoise(self):
        sampledMatrix = [len(ax) for ax in self.kAxes]
        self.noise = np.random.normal(0, self.noiseStd, sampledMatrix) + 1j * np.random.normal(0, self.noiseStd, sampledMatrix)


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
        self.measuredkspace = self.noise.copy()
        for component in self.kspaceComps:
            if 'Fat' in component:
                tissue = component[:component.find('Fat')]
                resonance = component[component.find('Fat'):]
                ratio = FATRESONANCES[resonance]['ratioWithFatSat' if self.FatSat else 'ratio']
                ratio *= TISSUES[tissue]['FF']
                self.measuredkspace += self.kspaceComps[component] * self.PDandT1w[resonance] * ratio
            else:
                if 'Water' in component:
                    tissue = component[:component.find('Water')]
                    ratio = 1 - TISSUES[tissue]['FF']    
                else:
                    tissue = component
                    ratio = 1.0
                self.measuredkspace += self.kspaceComps[component] * self.PDandT1w[tissue] * ratio
        self.updateSNR(self.decayedSignal * np.abs(self.PDandT1w[self.phantom['referenceTissue']]))
        self.updateScantime()
        add_to_pipeline(self.sequencePlotPipeline, ['renderSignalBoard'])


        self.griddedkspace = self.measuredkspace.copy() # For Cartesian       


    
    def partialFourierRecon(self):
        nBlank = self.oversampledMatrix[self.phaseDir] - self.oversampledPartialMatrix
        if (nBlank == 0):
            self.param.homodyne.precedence = -1
            self.fullkspace = np.copy(self.griddedkspace)
            return
        self.param.homodyne.precedence = 1
        shapeUnsampled = tuple(nBlank if dim==self.phaseDir else n for dim, n in enumerate(self.griddedkspace.shape))
        self.fullkspace = np.append(self.griddedkspace, np.zeros(shapeUnsampled), axis=self.phaseDir) # zerofill
        if self.homodyne:
            self.fullkspace *= homodyneWeights(self.oversampledMatrix[self.phaseDir], nBlank, self.phaseDir) # pre-weighting
            self.fullkspace += np.conjugate(np.flip(self.fullkspace))
    
    
    def apodization(self):
        self.apodizedkspace = self.fullkspace.copy()
        if not self.doApodize: 
            return
        self.apodizedkspace *= radialTukey(self.apodizationAlpha, self.oversampledMatrix)

    
    def zerofill(self):
        self.reconMatrix = [self.reconMatrixP, self.reconMatrixF] if self.doZerofill else [self.matrixP, self.matrixF]
        if self.freqDir==0:
            self.reconMatrix.reverse()
        self.oversampledReconMatrix = self.reconMatrix.copy()
        for dim in range(len(self.oversampledReconMatrix)):
            self.oversampledReconMatrix[dim] = int(np.round(self.reconMatrix[dim] * self.oversampledMatrix[dim] / self.matrix[dim]))
        self.zerofilledkspace = zerofill(np.fft.ifftshift(self.apodizedkspace), self.oversampledReconMatrix)
    
    
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
        add_to_pipeline(self.sequencePipeline, ['setupSliceSelection', 'setupPhasers', 'placeFatSat', 'updateMinTE', 'updateBWbounds', 'updateMatrixFbounds', 'updateFOVFbounds', 'updateMatrixPbounds', 'updateFOVPbounds', 'updateSliceThicknessBounds'])
        add_to_pipeline(self.sequencePlotPipeline, ['renderRFBoard'])


    def setupRefocusing(self):
        self.boards['RF']['objects']['refocusing'] = []
        if not isGradientEcho(self.sequence):
            for rf_echo in range(self.turboFactor):
                self.boards['RF']['objects']['refocusing'].append(sequence.getRF(flipAngle=180., dur=3., shape='hammingSinc',  name='refocusing {}'.format(rf_echo+1 if self.turboFactor>1 else '')))
            add_to_pipeline(self.sequencePipeline, ['placeRefocusing'])
        add_to_pipeline(self.sequencePipeline, ['setupSliceSelection', 'updateMinTE', 'updateMatrixPbounds', 'updateFOVPbounds', 'updateSliceThicknessBounds'])
        add_to_pipeline(self.sequencePlotPipeline, ['renderRFBoard'])



    def setupInversion(self):
        if self.sequence=='Inversion Recovery':
            self.boards['RF']['objects']['inversion'] = sequence.getRF(flipAngle=180., dur=3., shape='hammingSinc',  name='inversion')
            add_to_pipeline(self.sequencePipeline, ['placeInversion'])
        elif 'inversion' in self.boards['RF']['objects']:
            del self.boards['RF']['objects']['inversion']
        add_to_pipeline(self.sequencePipeline, ['setupSliceSelection', 'updateMaxTI', 'updateSliceThicknessBounds'])
        add_to_pipeline(self.sequencePlotPipeline, ['renderRFBoard'])
    
    
    def setupFatSat(self):
        if self.FatSat:
            self.boards['RF']['objects']['fatsat'] = sequence.getRF(flipAngle=90, time=0., dur=30./self.fieldStrength, shape='hammingSinc',  name='FatSat')
            spoilerArea = 30. # uTs/m
            self.boards['slice']['objects']['fatsat spoiler'] = sequence.getGradient('slice', totalArea=spoilerArea, name='FatSat spoiler')
        elif 'fatsat' in self.boards['RF']['objects']:
            del self.boards['RF']['objects']['fatsat']
            del self.boards['slice']['objects']['fatsat spoiler']
        add_to_pipeline(self.sequencePipeline, ['placeFatSat'])
        add_to_pipeline(self.sequencePlotPipeline, ['renderRFBoard', 'renderSliceBoard'])
    
    
    def setupSliceSelection(self):
        flatDur = self.boards['RF']['objects']['excitation']['dur_f']
        amp = self.boards['RF']['objects']['excitation']['FWHM_f'] / (self.sliceThickness * constants.GYRO)
        sliceSelectExcitation = sequence.getGradient('slice', 0., maxAmp=amp, flatDur=flatDur, name='slice select excitation')
        sliceRephaserArea = -sliceSelectExcitation['area_f']/2
        sliceSelectRephaser = sequence.getGradient('slice', totalArea=sliceRephaserArea, name='slice select rephaser')
        rephaserTime = (sliceSelectExcitation['dur_f'] + sliceSelectRephaser['dur_f']) / 2
        sequence.moveWaveform(sliceSelectRephaser, rephaserTime)
        self.boards['slice']['objects']['slice select excitation'] = sliceSelectExcitation
        self.boards['slice']['objects']['slice select rephaser'] = sliceSelectRephaser

        self.boards['slice']['objects']['slice select refocusing'] = []
        if not isGradientEcho(self.sequence):
            flatDur = self.boards['RF']['objects']['refocusing'][0]['dur_f']
            amp = self.boards['RF']['objects']['refocusing'][0]['FWHM_f'] / (self.sliceThickness * constants.GYRO)
            self.boards['slice']['objects']['slice select refocusing'] = []
            for rf_echo in range(self.turboFactor):
                self.boards['slice']['objects']['slice select refocusing'].append(sequence.getGradient('slice', maxAmp=amp, flatDur=flatDur, name='slice select refocusing'))
            add_to_pipeline(self.sequencePipeline, ['placeRefocusing'])
            
        if 'inversion' in self.boards['RF']['objects']:
            flatDur = self.boards['RF']['objects']['inversion']['dur_f']
            amp = self.boards['RF']['objects']['inversion']['FWHM_f'] / (self.inversionThkFactor * self.sliceThickness * constants.GYRO)
            self.boards['slice']['objects']['slice select inversion'] = sequence.getGradient('slice', maxAmp=amp, flatDur=flatDur, name='slice select inversion')

            spoilerArea = 30. # uTs/m
            self.boards['slice']['objects']['inversion spoiler'] = sequence.getGradient('slice', totalArea=spoilerArea, name='inversion spoiler')
            add_to_pipeline(self.sequencePipeline, ['placeInversion'])
        elif 'slice select inversion' in self.boards['slice']['objects']:
            del self.boards['slice']['objects']['slice select inversion']
            del self.boards['slice']['objects']['inversion spoiler']
        
        add_to_pipeline(self.sequencePipeline, ['setupPhasers', 'updateMinTE', 'updateMaxTI', 'updateMinTR', 'updateBWbounds'])
        add_to_pipeline(self.sequencePlotPipeline, ['renderSliceBoard', 'renderTRspan'])
    

    def setupReadouts(self):
        self.setup_frequency_encoding()
        
        flatArea = self.matrixF / (self.FOVF/1e3 * constants.GYRO) # uTs/m
        amp = self.pixelBandWidth * self.matrixF / (self.FOVF * constants.GYRO) # mT/m
        self.boards['frequency']['objects']['readouts'] = []
        self.boards['ADC']['objects']['samplings'] = []
        for rf_echo in range(self.turboFactor):
            gr_echoes = []
            samplings = []
            for gr_echo in range(self.EPIfactor):
                suffix = ' {}{}{}'.format(
                    rf_echo+1 if self.turboFactor>1 else '', 
                    '.' if (self.turboFactor>1 and self.EPIfactor>1) else '',
                    gr_echo+1 if self.EPIfactor>1 else '')
                readout = sequence.getGradient('frequency', maxAmp=amp, flatArea=flatArea, name='readout'+suffix)
                gr_echoes.append(readout)
                adc = sequence.getADC(dur=readout['flatDur_f'], name='sampling'+suffix)
                samplings.append(adc)
            self.boards['frequency']['objects']['readouts'].append(gr_echoes)
            self.boards['ADC']['objects']['samplings'].append(samplings)
        self.readout_risetime = readout['riseTime_f']
        prephaser = sequence.getGradient('frequency', totalArea=readout['area_f']/2, name='read prephaser')        
        self.boards['frequency']['objects']['read prephaser'] = prephaser
        add_to_pipeline(self.sequencePipeline, ['placeReadouts', 'setupPhasers', 'updateMinTE'])
    
    
    def setupPhasers(self):
        self.setup_phase_encoding()

        acq_FOVP = self.FOV[self.phaseDir]/self.matrix[self.phaseDir] * self.oversampledMatrix[self.phaseDir]
        phase_step_area = 1e3 / (acq_FOVP * constants.GYRO) # uTs/m
        maxPhaserArea = np.min(self.kAxes[self.phaseDir]) * 1e3 / constants.GYRO   # uTs/m
        
        self.maxPhaserDuration = sequence.getGradient('phase', totalArea=maxPhaserArea)['dur_f']

        self.max_blip_dur = 0
        if (self.EPIfactor > 1):
            max_blip_area = phase_step_area * self.num_shots * self.turboFactor
            self.max_blip_dur = sequence.getGradient('phase', totalArea=max_blip_area, name='dummy blip')['dur_f']
        readout = self.boards['frequency']['objects']['readouts'][0][0]
        readout_gap = max(self.max_blip_dur - 2 * readout['riseTime_f'], 0)
        self.gr_echo_spacing = readout['dur_f'] + readout_gap
        self.gre_echo_train_dur = self.EPIfactor * self.gr_echo_spacing - readout_gap
       
        self.set_readtrain_spacing()
        self.setup_phase_encoding_table()

        self.boards['phase']['objects']['phasers'] = []
        self.boards['phase']['objects']['rephasers'] = []
        self.boards['phase']['objects']['blips'] = []

        for rf_echo in range(self.turboFactor):
            phaserArea = maxPhaserArea + self.pe_table[self.shot-1][rf_echo][0] * phase_step_area
            suffix = ' {}'.format(rf_echo+1) if self.turboFactor>1 else ''
            phaser = sequence.getGradient('phase', totalArea=maxPhaserArea, name='phase encode'+suffix)
            sequence.rescaleGradient(phaser, phaserArea/maxPhaserArea)
            self.boards['phase']['objects']['phasers'].append(phaser)
            rephaserArea = -phaserArea
            blips = []
            for gr_echo in range(1, self.EPIfactor):
                blipArea = phase_step_area * (self.pe_table[self.shot-1][rf_echo][gr_echo]-self.pe_table[self.shot-1][rf_echo][gr_echo-1])
                blip = sequence.getGradient('phase', totalArea=blipArea, name='blip')
                blips.append(blip)
                rephaserArea -= blipArea
            self.boards['phase']['objects']['blips'].append(blips)
            rephaser = sequence.getGradient('phase', totalArea=maxPhaserArea, name='rephaser'+suffix)
            sequence.rescaleGradient(rephaser, rephaserArea/maxPhaserArea)
            self.boards['phase']['objects']['rephasers'].append(rephaser)
        add_to_pipeline(self.sequencePipeline, ['placePhasers', 'updateMinTE', 'updateBWbounds'])
    

    def setupSpoiler(self):
        spoilerArea = 30. # uTs/m
        self.boards['slice']['objects']['spoiler'] = sequence.getGradient('slice', totalArea=spoilerArea, name='spoiler')
        add_to_pipeline(self.sequencePipeline, ['placeSpoiler'])


    def get_readtrain_pos(self, rf_echo_num):
        # center position of gradient echo readout (train)
        return self.readtrain_spacing * (rf_echo_num + 1)
    
    
    def placeRefocusing(self):
        if not isGradientEcho(self.sequence):
            for rf_echo in range(self.turboFactor):
                pos = self.get_readtrain_pos(rf_echo) - self.readtrain_spacing/2
                sequence.moveWaveform(self.boards['RF']['objects']['refocusing'][rf_echo], pos)
                sequence.moveWaveform(self.boards['slice']['objects']['slice select refocusing'][rf_echo], pos)
            add_to_pipeline(self.sequencePipeline, ['updateBWbounds'])
            add_to_pipeline(self.sequencePlotPipeline, ['renderRFBoard', 'renderSliceBoard', 'calculate_k_trajectory'])
    

    def placeInversion(self):
        for board, name, renderer in [('RF', 'inversion', 'renderRFBoard'), ('slice', 'slice select inversion', 'renderSliceBoard')]:
            if name in self.boards[board]['objects']:
                sequence.moveWaveform(self.boards[board]['objects'][name], -self.TI)
                add_to_pipeline(self.sequencePlotPipeline, [renderer])
        if 'inversion spoiler' in self.boards['slice']['objects']:
            spoilerTime = self.boards['RF']['objects']['inversion']['time'][-1] + self.boards['slice']['objects']['inversion spoiler']['dur_f']/2
            sequence.moveWaveform(self.boards['slice']['objects']['inversion spoiler'], spoilerTime)
        add_to_pipeline(self.sequencePipeline, ['updateMinTR', 'updateBWbounds', 'updateSliceThicknessBounds'])
        add_to_pipeline(self.sequencePlotPipeline, ['renderSliceBoard', 'renderRFBoard', 'renderTRspan'])
    
    
    def placeFatSat(self):
        if 'fatsat' in self.boards['RF']['objects']:
            t = self.boards['slice']['objects']['slice select excitation']['time'][0] - self.boards['slice']['objects']['fatsat spoiler']['dur_f']/2
            sequence.moveWaveform(self.boards['slice']['objects']['fatsat spoiler'], t)
            t -= (self.boards['slice']['objects']['fatsat spoiler']['dur_f'] + self.boards['RF']['objects']['fatsat']['dur_f']) / 2
            sequence.moveWaveform(self.boards['RF']['objects']['fatsat'], t)
        add_to_pipeline(self.sequencePipeline, ['updateMinTR'])
        add_to_pipeline(self.sequencePlotPipeline, ['renderRFBoard', 'renderTRspan'])
    
    
    def placeReadouts(self):
        for rf_echo in range(self.turboFactor):
            readtrain_pos = self.get_readtrain_pos(rf_echo)
            for gr_echo in range(self.EPIfactor):
                pos = readtrain_pos + (gr_echo - (self.EPIfactor-1) / 2) * self.gr_echo_spacing
                for object in [self.boards['frequency']['objects']['readouts'], self.boards['ADC']['objects']['samplings']]:
                    sequence.moveWaveform(object[rf_echo][gr_echo], pos)
                if gr_echo%2 and self.boards['frequency']['objects']['readouts'][rf_echo][gr_echo]['area_f'] > 0:
                    sequence.rescaleGradient(self.boards['frequency']['objects']['readouts'][rf_echo][gr_echo], -1)
        if isGradientEcho(self.sequence):
            if self.boards['frequency']['objects']['read prephaser']['area_f'] > 0:
                sequence.rescaleGradient(self.boards['frequency']['objects']['read prephaser'], -1)
            firstReadout = self.boards['frequency']['objects']['readouts'][0][0]
            prephaseTime = firstReadout['center_f'] - sum([grad['dur_f'] for grad in [self.boards['frequency']['objects']['read prephaser'], firstReadout]])/2
        else:
            if self.boards['frequency']['objects']['read prephaser']['area_f'] < 0:
                sequence.rescaleGradient(self.boards['frequency']['objects']['read prephaser'], -1)
            prephaseTime = sum([self.boards[b]['objects'][name]['dur_f'] for (b, name) in [('RF', 'excitation'), ('frequency', 'read prephaser')]])/2
        sequence.moveWaveform(self.boards['frequency']['objects']['read prephaser'], prephaseTime)
        add_to_pipeline(self.sequencePipeline, ['placePhasers', 'placeSpoiler'])
        add_to_pipeline(self.sequencePlotPipeline, ['renderFrequencyBoard', 'renderRFBoard'])
    
    
    def placePhasers(self):
        for rf_echo in range(self.turboFactor):
            readtrain_pos = self.get_readtrain_pos(rf_echo)
            
            phaserDur = self.boards['phase']['objects']['phasers'][rf_echo]['dur_f']
            phaserTime = readtrain_pos - (self.gre_echo_train_dur + phaserDur)/2 + self.readout_risetime
            sequence.moveWaveform(self.boards['phase']['objects']['phasers'][rf_echo], phaserTime)

            for gr_echo in range(self.EPIfactor-1):
                blipTime = readtrain_pos + self.gr_echo_spacing * (gr_echo - self.EPIfactor/2 + 1)
                sequence.moveWaveform(self.boards['phase']['objects']['blips'][rf_echo][gr_echo], blipTime)

            rephaserDur = self.boards['phase']['objects']['rephasers'][rf_echo]['dur_f']
            rephaserTime = readtrain_pos + (self.gre_echo_train_dur + rephaserDur)/2 - self.readout_risetime
            sequence.moveWaveform(self.boards['phase']['objects']['rephasers'][rf_echo], rephaserTime)
        
        add_to_pipeline(self.sequencePlotPipeline, ['renderPhaseBoard'])


    def placeSpoiler(self):
        spoilerTime = self.boards['frequency']['objects']['readouts'][-1][-1]['center_f'] + (self.boards['frequency']['objects']['readouts'][-1][-1]['flatDur_f'] + self.boards['slice']['objects']['spoiler']['dur_f']) / 2
        sequence.moveWaveform(self.boards['slice']['objects']['spoiler'], spoilerTime)
        add_to_pipeline(self.sequencePipeline, ['updateMinTR', 'updateSliceThicknessBounds'])
        add_to_pipeline(self.sequencePlotPipeline, ['renderSliceBoard'])
    

    def addSignals(self):
        self.boards['signal']['objects']['signals'] = []
        scale = 1/np.max(np.abs(np.real(self.measuredkspace)))
        signalExponent = .5
        for rf_echo in range(self.turboFactor):
            signals = []
            for gr_echo in range(self.EPIfactor):
                ky = self.pe_table[self.shot-1][rf_echo][gr_echo]
                waveform = np.real(np.take(self.measuredkspace, indices=ky, axis=self.phaseDir))
                t = np.take(self.timeAfterExcitation, indices=ky, axis=self.phaseDir)
                signal = sequence.getSignal(waveform, t, scale, signalExponent)
                signals.append(signal)
            self.boards['signal']['objects']['signals'].append(signals)
    

    def update_k_line_coords(self, attr, old, hoverIndex):
        if len(hoverIndex['index']) > 0:
            object = self.boards[hoverIndex['board'][0]]['object_list'][hoverIndex['index'][0]]
            self.kLine.event(coords=list(self.get_k_on_interval(object['time'][[0, -1]])))
        else:
            self.kLine.event(coords=[None])
    
    
    def getHoverTool(self, board, obj):
        attributes = [attr for attr in obj.keys() if attr not in ['time', board] and '_f' not in attr]
        if board in ['frequency', 'phase', 'RF', 'ADC']:
            with open(Path(__file__).parent / 'hoverCallback.js', 'r') as file:
                hoverCallback = CustomJS(args={'hoverIndex': self.hoverIndex, 'board': board}, code=file.read())
        else:
            hoverCallback = None
        hover = HoverTool(tooltips=[(attr, '@{}'.format(attr)) for attr in attributes], attachment='below', callback=hoverCallback)
        return hover, attributes
    

    def renderPolygons(self, board, hoverTool=True):
        if self.boards[board]['objects']:
            object_list = flatten_dicts(self.boards[board]['objects'].values())
            self.boards[board]['object_list'] = object_list
            self.boardPlots[board]['area'] = hv.Area(sequence.accumulateWaveforms(object_list, board), self.timeDim, self.boards[board]['dim']).opts(color=self.boards[board]['color'])
            hover, attributes = self.getHoverTool(board, object_list[0])
            tools = [hover] if hoverTool else []
            self.boardPlots[board]['polygons'] = hv.Polygons(object_list, kdims=[self.timeDim, self.boards[board]['dim']], vdims=attributes).opts(tools=tools, cmap=[self.boards[board]['color']], hooks=[hideframe_hook, partial(bounds_hook, xbounds=(-19000, 19000))])
    
    
    def renderADCrectangles(self, hoverTool=True):
        object_list = flatten_dicts(self.boards['ADC']['objects'].values())
        self.boards['ADC']['object_list'] = object_list
        hover, attributes = self.getHoverTool('ADC', object_list[0])
        tools = [hover] if hoverTool else []
        for obj in object_list:
            obj.update({'c1': obj['time'][0], 'c2': -2, 'c3': obj['time'][-1], 'c4': 2})
        self.boardPlots['signal']['ADC'] = hv.Rectangles(object_list, kdims=['c1', 'c2', 'c3', 'c4'], vdims=attributes).opts(tools=tools)
    

    def renderTRspan(self):
        t0 = self.getSeqStart()
        for board in ['frequency', 'phase', 'slice', 'RF', 'signal']:
            self.boardPlots[board]['TRspan'] = hv.VSpan(-20000, t0, kdims=[self.timeDim, self.boards[board]['dim']]).opts(color='gray', fill_alpha=.3)
            self.boardPlots[board]['TRspan'] *= hv.VSpan(t0 + self.TR, 20000, kdims=[self.timeDim, self.boards[board]['dim']]).opts(color='gray', fill_alpha=.3)
    

    def renderFrequencyBoard(self):
        self.renderPolygons('frequency', hoverTool=True)
        add_to_pipeline(self.sequencePlotPipeline, ['calculate_k_trajectory'])
    

    def renderPhaseBoard(self):
        self.renderPolygons('phase', hoverTool=True)
        add_to_pipeline(self.sequencePlotPipeline, ['calculate_k_trajectory'])
    

    def renderSliceBoard(self):
        self.renderPolygons('slice', hoverTool=True)


    def renderRFBoard(self):
        self.renderPolygons('RF', hoverTool=True)
        add_to_pipeline(self.sequencePlotPipeline, ['calculate_k_trajectory'])
        

    def renderSignalBoard(self):
        self.addSignals()
        self.renderPolygons('signal', hoverTool=False)
        self.renderADCrectangles(hoverTool=True)
        add_to_pipeline(self.sequencePlotPipeline, ['calculate_k_trajectory'])
    
    
    def get_k_on_interval(self, interval):
        t = np.arange(*interval[[0, -1]], self.k_trajectory['dt'])
        kx = np.interp(t, self.k_trajectory['t'], self.k_trajectory['kx'])
        ky = np.interp(t, self.k_trajectory['t'], self.k_trajectory['ky'])
        return zip(kx, ky)
    

    def get_k_coords(self, t, gp, tp, refocus_intervals):
        g = np.interp(t, tp, gp)
        dk = np.diff(t) * (g[:-1] + np.diff(g)/2) * constants.GYRO * 1e-3
        k = np.insert(np.cumsum(dk), 0, 0.) # start at k=0
        for (ref_start, ref_stop) in refocus_intervals:
            # k inversion of refocusing pulse corresponds to negative shift of 2k:
            k_before = k[t<=ref_start][-1]
            refocus_times = t[(t>ref_start) & (t<ref_stop)]
            k[(t>ref_start) & (t<ref_stop)] -= 2 * k_before * (refocus_times - ref_start) / (ref_stop - ref_start)
            k[t>=ref_stop] -= 2 * k_before
        return k
    
    
    def calculate_k_trajectory(self):
        dt = .01
        refocus_intervals = [list(rf['time'][[0, -1]]) for rf in self.boards['RF']['objects']['refocusing']]
        t = np.concatenate((*(self.boardPlots[board]['area']['time'] for board in ['frequency', 'phase']), [t for ref in refocus_intervals for t in ref])) # k event times
        t = np.unique(np.concatenate((t, np.arange(0., max(t), dt)))) # merge with time grid
        kx = self.get_k_coords(t, *(self.boardPlots['frequency']['area'][dim] for dim in ['G read', 'time']), refocus_intervals)
        ky = self.get_k_coords(t, *(self.boardPlots['phase']['area'][dim] for dim in ['G phase', 'time']), refocus_intervals)
        if self.phaseDir==1:
            kx, ky = ky, kx
        self.k_trajectory = {'kx': kx, 'ky': ky, 't': t, 'dt': dt}
    

    @param.depends('sequence', 'FatSat', 'TR', 'TE', 'FA', 'TI', 'FOVF', 'FOVP', 'phaseOversampling', 'matrixF', 'matrixP', 'sliceThickness', 'trajectory', 'frequencyDirection', 'pixelBandWidth', 'partialFourier', 'turboFactor', 'EPIfactor', 'shot')
    def getSequencePlot(self):
        self.runSequencePlotPipeline()
        last = len(self.boardPlots)-1
        self.seqPlot = hv.Layout(list([hv.Overlay(list(boardPlot.values())).opts(width=1700, height=180 if n==last else 120, border=0, xaxis='bottom' if n==last else None) for n, boardPlot in enumerate(self.boardPlots.values())])).cols(1).options(toolbar='below')
        return self.seqPlot
    
    
    @param.depends('object', 'fieldStrength', 'sequence', 'FatSat', 'TR', 'TE', 'FA', 'TI', 'FOVF', 'FOVP', 'phaseOversampling', 'matrixF', 'matrixP', 'reconMatrixF', 'reconMatrixP', 'sliceThickness', 'trajectory', 'frequencyDirection', 'pixelBandWidth', 'NSA', 'partialFourier', 'turboFactor', 'EPIfactor', 'showProcessedKspace', 'kspaceExponent', 'homodyne', 'doApodize', 'apodizationAlpha', 'doZerofill')
    def getKspace(self):
        if self.showProcessedKspace:
            self.runReconPipeline()
            kAxes = []
            for dim in range(2):
                kAxes.append(getKaxis(self.oversampledReconMatrix[dim], self.FOV[dim]/self.reconMatrix[dim]))
                # half-sample shift axis when odd number of zeroes:
                if (self.oversampledReconMatrix[dim]-self.oversampledMatrix[dim])%2:
                    shift = self.reconMatrix[dim] / (2 * self.oversampledReconMatrix[dim] * self.FOV[dim])
                    kAxes[-1] += shift * (-1)**(self.oversampledMatrix[dim]%2)
            ksp = xr.DataArray(
                np.abs(np.fft.fftshift(self.zerofilledkspace))**self.kspaceExponent, 
                dims=('ky', 'kx'),
                coords={'kx': kAxes[1], 'ky': kAxes[0]}
            )
        else:
            self.runAcquisitionPipeline()
            ksp = xr.DataArray(
                np.abs(self.griddedkspace)**self.kspaceExponent, 
                dims=('ky', 'kx'),
                coords={'kx': self.kAxes[1], 'ky': self.kAxes[0]}
            )
        ksp.kx.attrs['units'] = ksp.ky.attrs['units'] = '1/mm'
        lim = 1.12 * max(self.kAxes[1])
        self.kspaceimage = hv.Image(ksp, vdims=['magnitude']).opts(xlim=(-lim,lim), ylim=(-lim,lim))
        return self.kspaceimage


    def getFOVbox(self):
        acqFOV = self.FOV.copy()
        acqFOV[self.phaseDir] *= self.oversampledMatrix[self.phaseDir] / self.matrix[self.phaseDir]
        return hv.Box(0, 0, tuple(acqFOV[::-1])).opts(color='lightblue') * hv.Box(0, 0, tuple(self.FOV[::-1])).opts(color='yellow')


    @param.depends('object', 'fieldStrength', 'sequence', 'FatSat', 'TR', 'TE', 'FA', 'TI', 'FOVF', 'FOVP', 'phaseOversampling', 'matrixF', 'matrixP', 'reconMatrixF', 'reconMatrixP', 'sliceThickness', 'trajectory', 'frequencyDirection', 'pixelBandWidth', 'NSA', 'partialFourier', 'turboFactor', 'EPIfactor', 'showFOV', 'homodyne', 'doApodize', 'apodizationAlpha', 'doZerofill')
    def getImage(self):
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


def hideShowButtonCallback(pane, event):
    if 'Show' in event.obj.name:
        pane.visible = True
        event.obj.name = event.obj.name.replace('Show', 'Hide')
    elif 'Hide' in event.obj.name:
        pane.visible = False
        event.obj.name = event.obj.name.replace('Hide', 'Show')


def loadButtonCallback(simulator, settingsFile, event):
    print('Loading settings from file', settingsFile)
    with open(settingsFile, 'r') as f:
        settings = toml.load(f)
    simulator.setParams(settings)


def saveButtonCallback(simulator, settingsFile, event):
    print('Saving settings to file', settingsFile)
    settings = simulator.getParams()
    with open(settingsFile, 'w') as f:
        toml.dump(settings, f)


def infoNumber(name, value, format, textColor):
    return pn.indicators.Number(default_color=textColor, name=name, format=format, font_size='12pt', title_size='12pt', value=value)


def getApp(darkMode=True, settingsFilestem=''):
    pn.config.theme = 'dark' if darkMode else 'default'
    textColor = 'white' if darkMode else 'black' # needed for pn.indicators.Number which doesn't respect pn.config.theme

    settingsFile = Path(settingsFilestem).with_suffix('.toml') if bool(settingsFilestem) else Path('')

    simulator = MRIsimulator(name='')
    title = '# SpinSight MRI simulator'
    author = '*Written by [Johan Berglund](mailto:johan.berglund@akademiska.se), Ph.D.*'
    try:
        version = '(v {})'.format(toml.load(Path(__file__).parent.parent/'pyproject.toml')['tool']['poetry']['version'])
    except (FileNotFoundError, KeyError):
        version = ''
    settingsParams = pn.panel(simulator.param, parameters=['object', 'fieldStrength', 'parameterStyle'], name='Settings')
    contrastParams = pn.panel(simulator.param, parameters=['FatSat', 'TR', 'TE', 'FA', 'TI'], widgets={'TR': pn.widgets.DiscreteSlider, 'TE': pn.widgets.DiscreteSlider, 'TI': pn.widgets.DiscreteSlider}, name='Contrast')
    geometryParams = pn.panel(simulator.param, parameters=['trajectory', 'frequencyDirection', 'FOVF', 'FOVP', 'phaseOversampling', 'voxelF', 'voxelP', 'matrixF', 'matrixP', 'reconVoxelF', 'reconVoxelP', 'reconMatrixF', 'reconMatrixP', 'sliceThickness'], widgets={'matrixF': pn.widgets.DiscreteSlider, 'matrixP': pn.widgets.DiscreteSlider, 'reconMatrixF': pn.widgets.DiscreteSlider, 'reconMatrixP': pn.widgets.DiscreteSlider, 'voxelF': pn.widgets.DiscreteSlider, 'voxelP': pn.widgets.DiscreteSlider, 'reconVoxelF': pn.widgets.DiscreteSlider, 'reconVoxelP': pn.widgets.DiscreteSlider}, name='Geometry')
    sequenceParams = pn.panel(simulator.param, parameters=['sequence', 'pixelBandWidth', 'FOVbandwidth', 'FWshift', 'NSA', 'partialFourier', 'turboFactor', 'EPIfactor'], widgets={'pixelBandWidth': pn.widgets.DiscreteSlider, 'FOVbandwidth': pn.widgets.DiscreteSlider, 'FWshift': pn.widgets.DiscreteSlider, 'EPIfactor': pn.widgets.DiscreteSlider}, name='Sequence')
    postprocParams = pn.panel(simulator.param, parameters=['homodyne', 'doApodize', 'apodizationAlpha', 'doZerofill'], name='Post-processing')

    infoPane = pn.Row(infoNumber(name='Relative SNR', format='{value:.0f}%', value=simulator.param.relativeSNR, textColor=textColor),
                      infoNumber(name='Scan time', format=('{value:.1f} sec'), value=simulator.param.scantime, textColor=textColor),
                      infoNumber(name='Fat/water shift', format='{value:.2f} pixels', value=simulator.param.FWshift, textColor=textColor),
                      infoNumber(name='Bandwidth', format='{value:.0f} Hz/pixel', value=simulator.param.pixelBandWidth, textColor=textColor))
    
    dmapKspace = pn.Column(hv.DynamicMap(simulator.getKspace) * simulator.kLine, 
                           pn.Row(simulator.param.showProcessedKspace, simulator.param.kspaceExponent), 
                           visible=False)
    dmapMRimage = hv.DynamicMap(simulator.getImage)
    dmapSequence = pn.Column(hv.DynamicMap(simulator.getSequencePlot), simulator.param.shot, visible=False)
    loadButton = pn.widgets.Button(name='Load settings', visible=settingsFile.is_file())
    loadButton.on_click(partial(loadButtonCallback, simulator, settingsFile))
    saveButton = pn.widgets.Button(name='Save settings', visible=settingsFile.is_file())
    saveButton.on_click(partial(saveButtonCallback, simulator, settingsFile))
    sequenceButton = pn.widgets.Button(name='Show sequence')
    sequenceButton.on_click(partial(hideShowButtonCallback, dmapSequence))
    kSpaceButton = pn.widgets.Button(name='Show k-space')
    kSpaceButton.on_click(partial(hideShowButtonCallback, dmapKspace))
    resetSNRbutton = pn.widgets.Button(name='Set reference SNR')
    resetSNRbutton.on_click(simulator.setReferenceSNR)
    dashboard = pn.Column(
        pn.Row(
            pn.Column(
                pn.pane.Markdown(title), 
                pn.Row(
                    pn.Column(
                        pn.Row(loadButton, saveButton), 
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
                    pn.Row(resetSNRbutton, simulator.param.showFOV), 
                    infoPane
                )
            ), 
            pn.Column(
                dmapKspace,
                postprocParams
            )
        ), 
        dmapSequence, 
        pn.Row(pn.pane.Markdown(author), pn.pane.Markdown(version))
    )
    return dashboard