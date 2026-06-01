import holoviews as hv
from holoviews import streams
import param
import numpy as np
import math
from pathlib import Path
import xarray as xr
from spinsight import constants, sequence, recon, phantom
from spinsight.DAG import build_graph
from bokeh.models import HoverTool, CustomJS, ColumnDataSource
from functools import partial
import warnings

hv.extension('bokeh')


BOARD_COLORS = {
    'frequency': 'cadetblue',
    'phase': 'cadetblue',
    'slice': 'cadetblue',
    'RF': 'red',
    'signal': 'orange',
    'ADC': 'peru',
}

def hline(time_dim, amp_dim):
        return hv.HLine(0.0, kdims=[time_dim, amp_dim]).opts(tools=['xwheel_zoom', 'xpan', 'reset'], default_tools=[], active_tools=['xwheel_zoom', 'xpan'])

def pixel_BW_to_shift(pixel_BW, B0=1.5):
    ''' Get fat/water chemical shift [pixels] from pixel bandwidth [Hz/pixel] and B0 [T]'''
    return np.abs(constants.FAT_RESONANCES['Fat2']['shift'] * constants.GYRO * B0 / pixel_BW)


def shift_to_pixel_BW(shift, B0=1.5):
    ''' Get pixel bandwidth [Hz/pixel] from fat/water chemical shift [pixels] and B0 [T]'''
    return np.abs(constants.FAT_RESONANCES['Fat2']['shift'] * constants.GYRO * B0 / shift)


def pixel_BW_to_FOV_BW(pixel_BW, matrix_F):
    ''' Get FOV bandwidth [±kHz] from pixel bandwidth [Hz/pixel] and read direction matrix'''
    return pixel_BW * matrix_F / 2e3


def FOV_BW_to_pixel_BW(FOV_BW, matrix_F):
    ''' Get pixel bandwidth [Hz/pixel] from FOV bandwidth [±kHz] and read direction matrix'''
    return FOV_BW / matrix_F * 2e3


def get_T2w(component, time_after_excitation, time_relative_inphase, B0):
    T2 = constants.TISSUES[component]['T2'][B0] if 'Fat' not in component else constants.FAT_RESONANCES[component]['T2'][B0]
    T2prim = 35. # ad hoc value [msec]
    E2 = np.exp(-np.abs(time_after_excitation)/T2)
    E2prim = np.exp(-np.abs(time_relative_inphase)/T2prim)
    return E2 * E2prim


def get_PD_and_T1w(component, sequence_type, TR, TE, TI, FA, B0):
    PD = constants.TISSUES[component]['PD'] if 'Fat' not in component else constants.FAT_RESONANCES[component]['PD']
    T1 = constants.TISSUES[component]['T1'][B0] if 'Fat' not in component else constants.FAT_RESONANCES[component]['T1'][B0]

    E1 = np.exp(-TR/T1)
    if sequence_type == 'Spin Echo':
        return PD * (1 - E1)
    elif sequence_type == 'Spoiled Gradient Echo':
        return PD * np.sin(np.radians(FA)) * (1 - E1) / (1 - np.cos(np.radians(FA)) * E1)
    elif sequence_type == 'Inversion Recovery':
        return PD * (1 - 2 * np.exp(-TI/T1) + E1)
    else:
        raise ValueError(f'Unknown sequence type: {sequence_type}')


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

    if c >= N - split_center:
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


def get_readtrain_pos(readtrain_spacing, rf_echo_num):
    # center position of gradient echo readout (train)
    return readtrain_spacing * (rf_echo_num + 1)


def readtrain_shift(gr_echo_spacing, centermost_gr_echo, num_gr_echoes):
    return gr_echo_spacing * (centermost_gr_echo - (num_gr_echoes-1)/2)


def TE_from_centermost_echoes(readtrain_spacing, centermost_rf_echo, gr_echo_spacing, centermost_gr_echo, EPI_factor):
    TE = readtrain_spacing * (1 + centermost_rf_echo)
    TE += readtrain_shift(gr_echo_spacing, centermost_gr_echo, EPI_factor)
    return TE


def get_k_coords(t, gp, tp, refocus_intervals):
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


def get_k_on_interval(interval, k_trajectory):
    t = np.arange(*interval[[0, -1]], k_trajectory['dt'])
    kx = np.interp(t, k_trajectory['t'], k_trajectory['kx'])
    ky = np.interp(t, k_trajectory['t'], k_trajectory['ky'])
    return zip(kx, ky)


def bounds_hook(plot, elem, xbounds=None):
    x_range = plot.handles['plot'].x_range
    if xbounds is not None:
        x_range.bounds = xbounds
    else:
        x_range.bounds = x_range.start, x_range.end 


def hideframe_hook(plot, elem):
    plot.handles['plot'].outline_line_color = None


def flatten_dicts(list_of_dicts_and_lists):
    if list_of_dicts_and_lists is None:
        return []
    res = []
    for v in list_of_dicts_and_lists:
        res += flatten_dicts(v) if isinstance(v, list) else [v]
    return res


def format_float(value, sigfigs=2):
    rounded = float(f'{value:.{sigfigs}g}')
    integer, decimal = str(rounded).split('.')
    exponent = int(math.floor(math.log10(abs(rounded))))
    num_decimals = sigfigs - exponent - 1
    if num_decimals <= 0:
        return integer
    decimal += '0' * (num_decimals - len(decimal))
    return '.'.join((integer, decimal))


def format_scantime(milliseconds):
    total_seconds = milliseconds / 1000
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)

    if minutes > 0:
        return f'{minutes} min {seconds} sec'
    elif seconds >= 10:
        return f'{seconds} sec'
    elif seconds > 0:
        return f'{total_seconds:.1f} sec'
    else:
        return f'{int(milliseconds)} msec'


param_values = {
    'TR': {tr + ' msec': float(tr) for tr in [format_float(tr, 2) for tr in 10.**np.linspace(0, 4, 500)]},
    'TE': {te + ' msec': float(te) for te in [format_float(te, 2) for te in 10.**np.linspace(0, 3, 500)]},
    'FA': {str(int(fa)) + '°': float(fa) for fa in range(1, 91)},
    'TI': {ti + ' msec': float(ti) for ti in [format_float(ti, 2) for ti in 10.**np.linspace(1.6, 4, 500)]},
    'FOV_P': {str(int(fov)) + ' mm': float(fov) for fov in range(100, 601)},
    'FOV_F': {str(int(fov)) + ' mm': float(fov) for fov in range(100, 601)},
    'phase_oversampling': {str(int(po)) + '%': float(po) for po in range(0, 101)},
    'pixel_bandwidth': {bw + ' Hz': float(bw) for bw in [format_float(bw, 3) for bw in 10.**np.linspace(2.097, 3.301, 500)]},
    'matrix_P': list(range(16, 600+1)),
    'matrix_F': list(range(16, 600+1)),
    'recon_matrix_P': list(range(16, 1200+1)),
    'recon_matrix_F': list(range(16, 1200+1)),
    'slice_thickness': {thk + ' mm': float(thk) for thk in [format_float(thk, 2) for thk in np.linspace(0.5, 10, 96)]},
    'EPI_factor': list(range(1, 64+1)),
}


class MRIsimulator(param.Parameterized):
    object = param.ObjectSelector(default='brain', label='Phantom object')
    field_strength = param.ObjectSelector(default=1.5, label='B0 field strength [T]')
    parameter_style = param.ObjectSelector(default='Matrix and Pixel BW', label='Parameter Style')
    min_voxel_size = param.Number(default=0.5) # [mm] limit on phantom resolution (to limit computation time)
    
    FatSat = param.Boolean(default=False, label='Fat saturation')
    TR = param.Selector(default=10000, label='TR')
    TE = param.Selector(default=10, label='TE')
    FA = param.Selector(default=90, precedence=-1, label='Flip angle')
    TI = param.Selector(default=40, precedence=-1, label='TI')
    
    trajectory = param.ObjectSelector(default=constants.TRAJECTORIES[0], precedence=1, label='k-space trajectory')
    frequency_direction = param.ObjectSelector(default=list(constants.DIRECTIONS.keys())[0], precedence=1, label='Frequency encoding direction')
    FOV_P = param.Selector(default=240, precedence=2, label='FOV x')
    FOV_F = param.Selector(default=240, precedence=2, label='FOV y')
    phase_oversampling = param.Selector(default=0, precedence=3, label='Phase oversampling')
    radial_factor = param.Number(default=1., precedence=-3, label='Spoke sampling factor')
    num_shots = param.Integer(precedence=-3)
    shot_label = param.String('shot')
    voxel_P = param.Selector(default=1.333, precedence=-4, label='Voxel size x')
    voxel_F = param.Selector(default=1.333, precedence=-4, label='Voxel size y')
    matrix_P = param.Selector(default=180, precedence=4, label='Acquisition matrix x')
    matrix_F = param.Selector(default=180, precedence=4, label='Acquisition matrix y')
    recon_voxel_P = param.Selector(default=0.666, precedence=-5, label='Reconstructed voxel size x')
    recon_voxel_F = param.Selector(default=0.666, precedence=-5, label='Reconstructed voxel size y')
    recon_matrix_P = param.Selector(default=360, precedence=5, label='Reconstruction matrix x')
    recon_matrix_F = param.Selector(default=360, precedence=5, label='Reconstruction matrix y')
    slice_thickness = param.Selector(default=3, precedence=6, label='Slice thickness')
    radial_FOV_oversampling = param.Number(default=2, step=0.01, precedence=9, label='Radial FOV oversampling factor')
    
    sequence_type = param.ObjectSelector(default=constants.SEQUENCES[0], precedence=1, label='Pulse sequence')
    pixel_bandwidth = param.Selector(default=480, precedence=2, label='Pixel bandwidth')
    FOV_bandwidth = param.Selector(default=pixel_BW_to_FOV_BW(480, 180), precedence=-2, label='FOV bandwidth')
    FW_shift = param.Selector(default=pixel_BW_to_shift(480), precedence=-2, label='Fat/water shift')
    NSA = param.Integer(default=1, precedence=3, label='NSA')
    partial_Fourier = param.Number(default=1, step=0.01, precedence=5, label='Partial Fourier factor')
    turbo_factor = param.Integer(default=1, precedence=6, label='Turbo factor')
    EPI_factor = param.Selector(default=1, precedence=7, label='EPI factor')
    shot = param.Integer(default=1, label='Displayed shot')

    image_type = param.ObjectSelector(default='Magnitude', label='Image type')
    show_FOV = param.Boolean(default=False, label='Show FOV')
    noise_gain = param.Number(default=3.)
    reference_tissue = param.ObjectSelector(label='Reference tissue')
    SNR = param.Number(label='SNR')
    reference_SNR = param.Number(default=1, label='Reference SNR')
    relative_SNR = param.Number(label='Relative SNR [%]')
    scantime = param.String(label='Scan time')
    spoke_angle = param.Number(label='Spoke angle [°]')

    kspace_type = param.ObjectSelector(default='Magnitude', label='k-space type')
    show_processed_kspace = param.Boolean(default=False, label='Show processed k-space')
    kspace_exponent = param.Number(default=0.2, step=.01, label='k-space exponent')
    homodyne = param.Boolean(default=True, precedence=1, label='Homodyne')
    do_apodize = param.Boolean(default=True, precedence=2, label='Apodization')
    apodization_alpha = param.Number(default=0.25, step=.01, precedence=3, label='Apodization alpha')
    do_zerofill = param.Boolean(default=True, precedence=4, label='Zerofill')

    def __init__(self, **params):
        
        super().__init__(**params)

        self.init_bounds()

        def arrow(coords):
            angle = 0
            if len(coords)>1:
                angle = -np.degrees(math.atan2(coords[-1][0]-coords[-2][0], coords[-1][1]-coords[-2][1]))
            return hv.Curve(coords) * hv.Points(coords[-1]).opts(angle=angle, marker='triangle')
        
        arrow_stream = streams.Stream.define('arrow', coords=[None])
        self.k_line = hv.DynamicMap(arrow, streams=[arrow_stream()])

        self.hover_index = ColumnDataSource({'index': [], 'board': []})
        self.hover_index.on_change('data', self.update_k_line_coords)
        
        self.outbound_params = set()

        hv.opts.defaults(hv.opts.Image(width=500, height=500, invert_yaxis=False, toolbar='below', cmap='gray', aspect='equal'))
        hv.opts.defaults(hv.opts.HLine(line_width=1.5, line_color='gray'))
        hv.opts.defaults(hv.opts.VSpan(color='orange', fill_alpha=.1, hover_fill_alpha=.8, default_tools=[]))
        hv.opts.defaults(hv.opts.Rectangles(color=BOARD_COLORS['ADC'], line_color=BOARD_COLORS['ADC'], fill_alpha=.1, line_alpha=.3, hover_fill_alpha=.8, default_tools=[]))
        hv.opts.defaults(hv.opts.Box(line_width=3))
        hv.opts.defaults(hv.opts.Ellipse(line_width=3))
        hv.opts.defaults(hv.opts.Area(fill_alpha=.5, line_width=1.5, line_color='gray', default_tools=[]))
        hv.opts.defaults(hv.opts.Polygons(line_width=1.5, fill_alpha=0, line_alpha=0, line_color='gray', selection_line_color='black', hover_fill_alpha=.8, hover_line_alpha=1, selection_fill_alpha=.8, selection_line_alpha=1, nonselection_line_alpha=0, default_tools=[]))
        hv.opts.defaults(hv.opts.Curve(line_width=5, line_color=BOARD_COLORS['ADC']))
        hv.opts.defaults(hv.opts.Points(line_color=None, color=BOARD_COLORS['ADC'], size=15))

        # constants
        self.max_amp = 25. # mT/m
        self.max_slew = 80. # T/m/s
        self.inversion_thk_factor = 1.1 # make inversion slice 10% thicker

        node_specs = {par: {'params': self} for par in self.param if par != 'name'}

        node_specs['set_min_TR'] = {
            'action': True,
            'func': self.set_min_TR,
            'parents': ['min_TR']
        }
        
        node_specs['set_TE_bounds'] = {
            'action': True,
            'func': self.set_TE_bounds,
            'parents': ['TR', 'min_TR', 'TE', 'min_TE']
        }
        
        node_specs['set_max_TI'] = {
            'action': True,
            'func': self.set_max_TI,
            'parents': ['sequence_type', 'TR', 'min_TR', 'TI']
        }
        
        node_specs['set_labels_by_trajectory'] = {
            'action': True,
            'func': self.set_labels_by_trajectory,
            'parents': ['shot_label']
        }
        
        node_specs['set_trajectory_objects'] = {
            'action': True,
            'func': self.set_trajectory_objects,
            'parents': ['EPI_factor', 'turbo_factor']
        }

        node_specs['set_BW_bounds'] = {
            'action': True,
            'func': self.set_BW_bounds,
            'parents': ['matrix_F', 'FOV_F', 'is_gradient_echo', 'gre_max_read_duration', 'RF_refocusing', 'turbo_factor', 'EPI_factor', 'TE', 'RF_excitation', 'readtrain_spacing', 'phaser_duration', 'max_blip_dur', 'slice_select_refocusing', 'TR', 'sequence_start', 'spoiler']
        }
        
        node_specs['set_matrix_F_bounds'] = {
            'action': True,
            'func': self.set_matrix_F_bounds,
            'parents': ['max_readout_area', 'FOV_F', 'parameter_style', 'FOV_bandwidth']
        }
        
        node_specs['set_matrix_P_bounds'] = {
            'action': True,
            'func': self.set_matrix_P_bounds,
            'parents': ['max_phaser_area', 'FOV_P']
        }
        
        node_specs['set_FOV_F_bounds'] = {
            'action': True,
            'func': self.set_FOV_F_bounds,
            'parents': ['matrix_F', 'max_readout_area', 'parameter_style', 'voxel_F', 'recon_voxel_F']
        }
        
        node_specs['set_FOV_P_bounds'] = {
            'action': True,
            'func': self.set_FOV_P_bounds,
            'parents': ['matrix_P', 'max_phaser_area', 'parameter_style', 'voxel_P', 'recon_voxel_P']
        }
        
        node_specs['set_FW_shift_objects'] = {
            'action': True,
            'func': self.set_FW_shift_objects,
            'parents': ['field_strength']
        }
        
        node_specs['set_FOV_bandwidth_objects'] = {
            'action': True,
            'func': self.set_FOV_bandwidth_objects,
            'parents': ['matrix_F']
        }
        
        node_specs['set_voxel_F_objects'] = {
            'action': True,
            'func': self.set_voxel_F_objects,
            'parents': ['FOV_F']
        }
        
        node_specs['set_voxel_P_objects'] = {
            'action': True,
            'func': self.set_voxel_P_objects,
            'parents': ['FOV_P']
        }
        
        node_specs['set_recon_voxel_F_objects'] = {
            'action': True,
            'func': self.set_recon_voxel_F_objects,
            'parents': ['FOV_F']
        }
        
        node_specs['set_recon_voxel_P_objects'] = {
            'action': True,
            'func': self.set_recon_voxel_P_objects,
            'parents': ['FOV_P']
        }
        
        node_specs['set_slice_thickness_bounds'] = {
            'action': True,
            'func': self.set_slice_thickness_bounds,
            'parents': ['RF_excitation', 'is_gradient_echo', 'RF_refocusing', 'sequence_type', 'RF_inversion', 'TR', 'spoiler', 'sampling_windows']
        }
        
        node_specs['set_turbo_factor_bounds'] = {
            'action': True,
            'func': self.set_turbo_factor_bounds,
            'parents': ['matrix', 'phase_dir', 'partial_Fourier', 'EPI_factor']
        }

        node_specs['set_EPI_factor_objects'] = {
            'action': True,
            'func': self.set_EPI_factor_objects,
            'parents': ['matrix', 'phase_dir', 'partial_Fourier', 'turbo_factor']
        }
        
        node_specs['set_homodyne_visibility'] = {
            'action': True,
            'func': self.set_homodyne_visibility,
            'parents': ['num_blank_lines', 'is_radial']
        }

        node_specs['set_reference_tissue_objects'] = {
            'action': True,
            'func': self.set_reference_tissue_objects,
            'parents': ['tissues']
        }
        
        node_specs['set_shot_bounds'] = {
            'action': True,
            'func': self.set_shot_bounds,
            'parents': ['num_shots']
        }
        
        node_specs['phantom'] = {
            'func': lambda object, min_voxel_size:
                    phantom.load(object, min_voxel_size),
            'parents': ['object', 'min_voxel_size']
        }
        
        node_specs['tissues'] = {
            'func': lambda fantom:
                    list(fantom['shapes'].keys()),
            'parents': ['phantom']
        }

        node_specs['is_radial'] = {
            'func': lambda trajectory: 
                    trajectory in ['Radial', 'PROPELLER'],
            'parents': ['trajectory']
        }
        
        node_specs['is_gradient_echo'] = {
            'func': lambda sequence_type: 
                    'Gradient Echo' in sequence_type, 
            'parents': ['sequence_type']
        }

        node_specs['freq_dir'] = {
            'func': lambda frequency_direction, is_radial: 
                    constants.DIRECTIONS[frequency_direction] if not is_radial else 1,
            'parents': ['frequency_direction', 'is_radial']
        }

        node_specs['phase_dir'] = {
            'func': lambda freq_dir: 
                    1 - freq_dir,
            'parents': ['freq_dir']
        }
        
        node_specs['FOV'] = {
            'func': lambda FOV_F, FOV_P, freq_dir: 
                    [FOV_P, FOV_F] if freq_dir else [FOV_F, FOV_P],
            'parents': ['FOV_F', 'FOV_P', 'freq_dir']
        }


        node_specs['matrix'] = {
            'func': lambda matrix_F, matrix_P, freq_dir: 
                    [matrix_P, matrix_F] if freq_dir else [matrix_F, matrix_P],
            'parents': ['matrix_F', 'matrix_P', 'freq_dir']
        }

        node_specs['recon_matrix'] = {
            'func': lambda recon_matrix_P, recon_matrix_F, freq_dir, do_zerofill, matrix:
                    ([recon_matrix_P, recon_matrix_F] if freq_dir else [recon_matrix_F, recon_matrix_P]) if do_zerofill else matrix,
            'parents': ['recon_matrix_P', 'recon_matrix_F', 'freq_dir', 'do_zerofill', 'matrix']
        }

        node_specs['RF_excitation'] = {
            'func': self.RF_excitation_func,
            'parents': ['FA', 'is_gradient_echo']
        }

        node_specs['RF_refocusing_floating'] = {
            'func': self.RF_refocusing_floating_func,
            'parents': ['is_gradient_echo', 'turbo_factor']
        }

        node_specs['RF_inversion_floating'] = {
            'func': self.RF_inversion_floating_func,
            'parents': ['sequence_type']
        }

        node_specs['RF_FatSat_floating'] = {
            'func': self.RF_FatSat_floating_func,
            'parents': ['FatSat', 'field_strength']
        }

        node_specs['FatSat_spoiler_floating'] = {
            'func': self.FatSat_spoiler_floating_func,
            'parents': ['FatSat']
        }

        node_specs['slice_select_excitation'] = {
            'func': self.slice_select_excitation_func,
            'parents': ['RF_excitation', 'slice_thickness']
        }

        node_specs['slice_select_rephaser'] = {
            'func': self.slice_select_rephaser_func,
            'parents': ['slice_select_excitation']
        }

        node_specs['slice_select_refocusing_floating'] = {
            'func': self.slice_select_refocusing_floating_func,
            'parents': ['RF_refocusing_floating', 'slice_thickness', 'turbo_factor']
        }

        node_specs['slice_select_inversion_floating'] = {
            'func': self.slice_select_inversion_floating_func,
            'parents': ['sequence_type', 'RF_inversion_floating', 'slice_thickness']
        }

        node_specs['inversion_spoiler_floating'] = {
            'func': self.inversion_spoiler_floating_func,
            'parents': ['sequence_type']
        }

        node_specs['readouts_floating'] = {
            'func': self.readouts_floating_func,
            'parents': ['k_read_axis', 'pixel_bandwidth', 'matrix_F', 'FOV_F', 'turbo_factor', 'EPI_factor']
        }

        node_specs['sampling_windows_floating'] = {
            'func': self.sampling_windows_floating_func,
            'parents': ['turbo_factor', 'EPI_factor', 'readouts_floating']
        }

        node_specs['readout_risetime'] = {
            'func': self.readout_risetime_func,
            'parents': ['readouts_floating']
        }

        node_specs['read_prephaser_floating'] = {
            'func': self.read_prephaser_floating_func,
            'parents': ['readouts_floating']
        }

        node_specs['phase_step_area'] = {
            'func': lambda k_phase_axis:
                    np.mean(np.diff(k_phase_axis)) * 1e3 / constants.GYRO, # uTs/m
            'parents': ['k_phase_axis']
        }
        
        node_specs['largest_phaser_area'] = {
            'func': lambda k_phase_axis:
                    np.min(k_phase_axis) * 1e3 / constants.GYRO, # uTs/m
            'parents': ['k_phase_axis']
        }
        
        node_specs['phaser_duration'] = {
            'func': self.phaser_duration_func,
            'parents': ['largest_phaser_area']
        }
        
        node_specs['max_blip_dur'] = {
            'func': self.max_blip_dur_func,
            'parents': ['EPI_factor', 'phase_step_area', 'num_shots', 'turbo_factor']
        }
        
        node_specs['readout_gap'] = {
            'func': lambda max_blip_dur, readouts:
                    max(max_blip_dur - 2 * readouts[0][0]['risetime_f'], 0),
            'parents': ['max_blip_dur', 'readouts_floating']
        }
        
        node_specs['gr_echo_spacing'] = {
            'func': lambda readouts, readout_gap:
                    readouts[0][0]['dur_f'] + readout_gap,
            'parents': ['readouts_floating', 'readout_gap']
        }
        
        node_specs['gre_echo_train_dur'] = {
            'func': lambda EPI_factor, gr_echo_spacing, readout_gap:
                    EPI_factor * gr_echo_spacing - readout_gap,
            'parents': ['EPI_factor', 'gr_echo_spacing', 'readout_gap']
        }
        
        node_specs['phasers_floating'] = {
            'func': self.phasers_floating_func,
            'parents': ['turbo_factor', 'largest_phaser_area', 'pe_table', 'phase_step_area', 'shot']
        }
        
        node_specs['blips_floating'] = {
            'func': self.blips_floating_func,
            'parents': ['turbo_factor', 'EPI_factor', 'phase_step_area', 'pe_table', 'shot']
        }
        
        node_specs['rephasers_floating'] = {
            'func': self.rephasers_floating_func,
            'parents': ['turbo_factor', 'phasers_floating', 'blips_floating', 'largest_phaser_area']
        }
        
        node_specs['spoiler_floating'] = {
            'func': self.spoiler_floating_func,
            'parents': []
        }
        
        node_specs['slice_select_refocusing'] = {
            'func': self.slice_select_refocusing_func,
            'parents': ['slice_select_refocusing_floating', 'readtrain_spacing']
        }
        
        node_specs['RF_refocusing'] = {
            'func': self.RF_refocusing_func,
            'parents': ['RF_refocusing_floating', 'readtrain_spacing']
        }
        
        node_specs['slice_select_inversion'] = {
            'func': self.slice_select_inversion_func,
            'parents': ['slice_select_inversion_floating', 'TI']
        }
        
        node_specs['RF_inversion'] = {
            'func': self.RF_inversion_func,
            'parents': ['RF_inversion_floating', 'TI']
        }
        
        node_specs['inversion_spoiler'] = {
            'func': self.inversion_spoiler_func,
            'parents': ['inversion_spoiler_floating', 'RF_inversion']
        }
        
        node_specs['FatSat_spoiler'] = {
            'func': self.FatSat_spoiler_func,
            'parents': ['FatSat_spoiler_floating', 'slice_select_excitation']
        }
        
        node_specs['RF_FatSat'] = {
            'func': self.RF_FatSat_func,
            'parents': ['RF_FatSat_floating', 'FatSat_spoiler_floating']
        }
        
        node_specs['readouts'] = {
            'func': self.readouts_func,
            'parents': ['turbo_factor', 'readtrain_spacing', 'EPI_factor', 'gr_echo_spacing', 'readouts_floating']
        }

        node_specs['sampling_windows'] = {
            'func': self.sampling_windows_func,
            'parents': ['turbo_factor', 'readtrain_spacing', 'EPI_factor', 'gr_echo_spacing', 'sampling_windows_floating']
        }

        node_specs['read_prephaser'] = {
            'func': self.read_prephaser_func,
            'parents': ['read_prephaser_floating', 'is_gradient_echo', 'readouts', 'RF_excitation']
        }

        node_specs['phasers'] = {
            'func': self.phasers_func,
            'parents': ['turbo_factor', 'readtrain_spacing', 'phasers_floating', 'gre_echo_train_dur', 'readout_risetime']
        }

        node_specs['rephasers'] = {
            'func': self.rephasers_func,
            'parents': ['turbo_factor', 'readtrain_spacing', 'gre_echo_train_dur', 'readout_risetime', 'rephasers_floating']
        }

        node_specs['rephasers'] = {
            'func': self.rephasers_func,
            'parents': ['turbo_factor', 'readtrain_spacing', 'gre_echo_train_dur', 'readout_risetime', 'rephasers_floating']
        }
        
        node_specs['blips'] = {
            'func': self.blips_func,
            'parents': ['turbo_factor', 'readtrain_spacing', 'EPI_factor', 'gr_echo_spacing', 'blips_floating']
        }
        
        node_specs['spoiler'] = {
            'func': self.spoiler_func,
            'parents': ['readouts', 'spoiler_floating']
        }
        
        node_specs['sequence_start'] = {
            'func': self.sequence_start_func,
            'parents': ['sequence_type', 'slice_select_inversion', 'RF_FatSat', 'slice_select_excitation']
        }

        node_specs['signal_curves'] = {
            'func': self.signal_curves_func,
            'parents': ['measured_kspace', 'shot', 'is_radial', 'turbo_factor', 'EPI_factor', 'pe_table', 'phase_dir', 'time_after_excitation']
        }

        node_specs['k_read_axis'] = {
            'func': self.k_read_axis_func,
            'parents': ['freq_dir', 'FOV', 'matrix', 'is_radial', 'phantom', 'radial_FOV_oversampling']
        }

        node_specs['reverse_linear_order'] = {
             # TODO: implement logic (pick forward or reverse order that minimizes readtrin_spacing while respecting minimum spacing and TR)
            'func': lambda: False,
            'parents': []
        }

        node_specs['min_TE'] = {
            'func': self.min_TE_func,
            'parents': ['EPI_factor', 'centermost_rf_echo', 'centermost_gr_echo', 'min_readtrain_spacing', 'gr_echo_spacing']
        }
        
        node_specs['min_TR'] = {
            'func': self.min_TR_func,
            'parents': ['spoiler', 'sequence_start']
        }
        
        node_specs['gre_read_start_to_kcenter'] = {
            'func': self.gre_read_start_to_kcenter_func,
            'parents': ['is_gradient_echo', 'phasers', 'rephasers', 'TE', 'RF_excitation', 'read_prephaser', 'readout_risetime', 'slice_select_excitation', 'slice_select_rephaser']
        }

        node_specs['gre_kcenter_to_read_end'] = {
            'func': self.gre_kcenter_to_read_end_func,
            'parents': ['is_gradient_echo', 'TR', 'sequence_start', 'spoiler', 'TE']
        }

        node_specs['gre_max_read_duration'] = {
            'func': self.gre_max_read_duration_func,
            'parents': ['gre_read_start_to_kcenter', 'gre_kcenter_to_read_end', 'centermost_gr_echo', 'readout_risetime', 'EPI_factor']
        }

        node_specs['max_readout_area'] = {
            'func': self.max_readout_area_func,
            'parents': ['pixel_bandwidth', 'is_gradient_echo', 'centermost_gr_echo', 'RF_excitation', 'phaser_duration', 'slice_select_excitation', 'slice_select_rephaser', 'max_blip_dur', 'readtrain_spacing', 'RF_refocusing_floating', 'EPI_factor']
        }

        node_specs['max_phaser_area'] = {
            'func': self.max_phaser_area_func,
            'parents': ['is_gradient_echo', 'readtrain_spacing', 'RF_excitation', 'gre_echo_train_dur', 'readout_risetime', 'RF_refocusing']
        }

        node_specs['min_readtrain_spacing'] = {
            'func': self.min_readtrain_spacing_func,
            'parents': ['is_gradient_echo', 'RF_excitation', 'gre_echo_train_dur', 'readout_risetime', 'read_prephaser_floating', 'phaser_duration', 'slice_select_excitation', 'slice_select_rephaser', 'RF_refocusing_floating', 'slice_select_refocusing_floating']
        }

        node_specs['centermost_gr_echo'] = {
            'func': self.centermost_gr_echo_func,
            'parents': ['central_segment', 'reverse_linear_order', 'num_segm', 'turbo_factor']
        }
        
        node_specs['centermost_rf_echo'] = {
            'func': self.centermost_rf_echo_func,
            'parents': ['is_gradient_echo', 'EPI_factor', 'central_segment', 'turbo_factor', 'TE', 'min_readtrain_spacing']
        }

        node_specs['readtrain_spacing'] = {
            'func': self.readtrain_spacing_func,
            'parents': ['EPI_factor', 'gr_echo_spacing', 'TE', 'centermost_gr_echo', 'centermost_rf_echo']
        }

        node_specs['num_blades'] = {
            'func': lambda is_radial, matrix, radial_factor, turbo_factor, EPI_factor:
                    int(np.ceil(max(matrix) * radial_factor / turbo_factor / EPI_factor * np.pi / 2)) if is_radial else 1,
            'parents': ['is_radial', 'matrix', 'radial_factor', 'turbo_factor', 'EPI_factor']
        }

        node_specs['k_angles'] = {
            'func': lambda num_blades: 
                    np.linspace(0, np.pi, num_blades, endpoint=False),
            'parents': ['num_blades']
        }

        node_specs['spoke_angle'] = {
            'params': self,
            'func': lambda k_angles, shot: 
                    np.degrees(k_angles[min(shot-1, len(k_angles)-1)]),
            'parents': ['k_angles', 'shot']
        }

        node_specs['num_shots'] = {
            'params': self,
            'func': lambda matrix_P, phase_oversampling, partial_Fourier, turbo_factor, EPI_factor, is_radial, num_blades:
                    int(np.ceil(matrix_P * (1 + phase_oversampling / 100) * partial_Fourier / turbo_factor / EPI_factor)) if not is_radial else num_blades,
            'parents': ['matrix_P', 'phase_oversampling', 'partial_Fourier', 'turbo_factor', 'EPI_factor', 'is_radial', 'num_blades']
        }

        node_specs['shot_label'] = {
            'params': self,
            'func': lambda is_radial, EPI_factor, turbo_factor:
                    'shot' if not is_radial else 'spoke' if (EPI_factor * turbo_factor == 1) else 'blade',
            'parents': ['is_radial', 'EPI_factor', 'turbo_factor']
        }

        node_specs['num_measured_lines'] = {
            # measured lines per blade
            'func': lambda turbo_factor, EPI_factor, num_shots, is_radial:
                    turbo_factor * EPI_factor * (num_shots if not is_radial else 1),
            'parents': ['turbo_factor', 'EPI_factor', 'num_shots', 'is_radial']
        }

        node_specs['k_phase_axis'] = {
            'func': self.k_phase_axis_func,
            'parents': ['is_radial', 'num_measured_lines', 'matrix', 'phase_dir', 'phase_oversampling', 'FOV']
        }

        node_specs['num_blank_lines'] = {
            'func': lambda k_phase_axis, lines_to_measure:
                    len(k_phase_axis) - sum(lines_to_measure),
            'parents': ['k_phase_axis', 'lines_to_measure']
        }

        node_specs['lines_to_measure'] = {
            'func': self.lines_to_measure_func,
            'parents': ['k_phase_axis', 'num_measured_lines']
        }

        node_specs['num_segm'] = {
            'func': self.num_segm_func,
            'parents': ['num_measured_lines', 'num_blades', 'num_shots']
        }

        node_specs['num_sym_lines'] = {
            'func': self.num_sym_lines_func,
            'parents': ['num_measured_lines', 'k_phase_axis']
        }

        node_specs['split_center'] = {
            'func': self.split_center_func,
            'parents': ['num_sym_lines', 'num_shots']
        }

        node_specs['num_sym_segm'] = {
            'func': self.num_sym_segm_func,
            'parents': ['split_center', 'num_sym_lines', 'num_blades', 'num_shots']
        }

        node_specs['central_segment'] = {
            'func': self.central_segment_func,
            'parents': ['num_segm', 'num_sym_segm']
        }

        node_specs['pe_table'] = {
            'func': self.pe_table_func,
            'parents': ['EPI_factor', 'turbo_factor', 'num_sym_segm', 'centermost_rf_echo', 'is_radial', 'num_shots', 'reverse_linear_order', 'lines_to_measure']
        }

        node_specs['signal_level'] = {
            'func': self.signal_level_func,
            'parents': ['k_read_axis', 'lines_to_measure', 'num_blades', 'slice_thickness', 'FOV', 'matrix']
        }

        node_specs['spin_echoes'] = {
            'func': self.spin_echoes_func,
            'parents': ['lines_to_measure', 'pe_table', 'readtrain_spacing']
        }

        node_specs['sampling_time'] = {
            'func': self.sampling_time_func,
            'parents': ['pixel_bandwidth', 'k_read_axis']
        }

        node_specs['time_after_excitation'] = {
            'func': self.time_after_excitation_func,
            'parents': ['lines_to_measure', 'pe_table', 'readouts', 'sampling_time', 'freq_dir', 'phase_dir']
        }

        node_specs['time_relative_inphase'] = {
            'func': self.time_relative_inphase_func,
            'parents': ['time_after_excitation', 'is_gradient_echo', 'spin_echoes', 'phase_dir']
        }

        node_specs['dephasing'] = {
            'func': self.dephasing_func,
            'parents': ['field_strength', 'time_relative_inphase']
        }

        node_specs['T2w'] = {
            'func': self.T2w_func,
            'parents': ['tissues', 'time_after_excitation', 'time_relative_inphase', 'field_strength']
        }

        node_specs['k_axes'] = {
            'func': self.k_axes_func,
            'parents': ['freq_dir', 'phase_dir', 'k_read_axis', 'k_phase_axis', 'lines_to_measure']
        }
        
        node_specs['k_grid_axes'] = {
            'func': self.k_grid_axes_func,
            'parents': ['is_radial', 'k_axes', 'FOV', 'matrix', 'phantom']
        }
        
        node_specs['k_samples'] = {
            'func': self.k_samples_func,
            'parents': ['k_axes', 'k_angles']
        }
        
        node_specs['plain_kspace_comps'] = {
            'func': self.plain_kspace_comps_func,
            'parents': ['is_radial', 'phantom', 'k_grid_axes', 'k_samples']
        }
        
        node_specs['thick_kspace_comps'] = {
            'func': self.thick_kspace_comps_func,
            'parents': ['slice_thickness', 'k_samples', 'plain_kspace_comps']
        }
        
        node_specs['kspace_comps'] = {
            'func': self.kspace_comps_func,
            'parents': ['tissues', 'thick_kspace_comps', 'T2w', 'dephasing']
        }
        
        node_specs['decayed_signal'] = {
            'func': self.decayed_signal_func,
            'parents': ['signal_level', 'T2w', 'reference_tissue', 'k_read_axis', 'k_phase_axis', 'freq_dir']
        }

        node_specs['PD_and_T1w'] = {
            'func': self.PD_and_T1w_func,
            'parents': ['sequence_type', 'TR', 'TE', 'TI', 'FA', 'field_strength', 'tissues']
        }

        node_specs['reference_signal'] = {
            'func': lambda decayed_signal, PD_and_T1w, reference_tissue:
                    decayed_signal * np.abs(PD_and_T1w[reference_tissue]),
            'parents': ['decayed_signal', 'PD_and_T1w', 'reference_tissue']
        }

        node_specs['noise_std'] = {
            'func': self.noise_std_func,
            'parents': ['sampling_time', 'noise_gain', 'NSA', 'field_strength']
        }

        node_specs['noise'] = {
            'func': self.noise_func,
            'parents': ['k_samples', 'noise_std']
        }

        node_specs['SNR'] = {
            'func': lambda signal, noise_std:
                    signal / noise_std,
            'parents': ['reference_signal', 'noise_std']
        }

        node_specs['relative_SNR'] = {
            'params': self,
            'func': lambda SNR, reference_SNR:
                    SNR / reference_SNR * 100,
            'parents': ['SNR', 'reference_SNR']
        }
        
        node_specs['FOV_box'] = {
            'func': self.FOV_box_func,
            'parents': ['show_FOV', 'is_radial', 'FOV', 'matrix', 'freq_dir', 'phase_dir', 'k_read_axis', 'k_phase_axis']
        }
        
        node_specs['scantime'] = {
            'params': self,
            'func': lambda num_shots, NSA, TR:
                    format_scantime(num_shots * NSA * TR),
            'parents': ['num_shots', 'NSA', 'TR']
        }
        
        node_specs['measured_kspace'] = {
            'func': self.measured_kspace_func,
            'parents': ['noise', 'kspace_comps', 'FatSat', 'PD_and_T1w']
        }
        
        node_specs['gridded_kspace'] = {
            'func': self.gridded_kspace_func,
            'parents': ['k_grid_axes', 'is_radial', 'measured_kspace', 'k_samples', 'FOV', 'matrix']
        }
        
        node_specs['full_kspace'] = {
            'func': self.full_kspace_func,
            'parents': ['num_blank_lines', 'is_radial', 'gridded_kspace', 'phase_dir', 'homodyne', 'k_phase_axis']
        }
        
        node_specs['full_k_matrix'] = {
            'func': lambda full_kspace: 
                    full_kspace.shape, 
            'parents': ['full_kspace']
        }
        
        node_specs['apodized_kspace'] = {
            'func': self.apodized_kspace_func,
            'parents': ['full_kspace', 'do_apodize', 'apodization_alpha']
        }
        
        node_specs['oversampled_recon_matrix'] = {
            'func': self.oversampled_recon_matrix_func,
            'parents': ['recon_matrix', 'full_k_matrix', 'matrix']
        }
        
        node_specs['k_trajectory'] = {
            'func': self.k_trajectory_func,
            'parents': ['RF_refocusing', 'frequency_board', 'phase_board', 'is_radial', 'phase_dir', 'spoke_angle']
        }

        node_specs['time_dim'] = {
            'func': self.time_dim_func,
            'parents': []
        }

        node_specs['frequency_dim'] = {
            'func': self.frequency_dim_func,
            'parents': []
        }

        node_specs['phase_dim'] = {
            'func': self.phase_dim_func,
            'parents': []
        }

        node_specs['slice_dim'] = {
            'func': self.slice_dim_func,
            'parents': []
        }

        node_specs['RF_dim'] = {
            'func': self.RF_dim_func,
            'parents': []
        }

        node_specs['signal_dim'] = {
            'func': self.signal_dim_func,
            'parents': []
        }

        node_specs['ADC_dim'] = {
            'func': self.ADC_dim_func,
            'parents': []
        }

        node_specs['frequency_objects'] = {
            'func': self.frequency_objects_func,
            'parents': ['read_prephaser', 'readouts']
        }

        node_specs['phase_objects'] = {
            'func': self.phase_objects_func,
            'parents': ['phasers', 'rephasers', 'blips']
        }
        
        node_specs['slice_objects'] = {
            'func': self.slice_objects_func,
            'parents': ['slice_select_inversion', 'inversion_spoiler', 'FatSat_spoiler', 'slice_select_excitation', 'slice_select_rephaser', 'slice_select_refocusing', 'spoiler']
        }
        
        node_specs['RF_objects'] = {
            'func': self.RF_objects_func,
            'parents': ['RF_inversion', 'RF_FatSat', 'RF_excitation', 'RF_refocusing']
        }
        
        node_specs['signal_objects'] = {
            'func': self.signal_objects_func,
            'parents': ['signal_curves']
        }
        
        node_specs['ADC_objects'] = {
            'func': self.ADC_objects_func,
            'parents': ['sampling_windows']
        }
        
        node_specs['TR_span'] = {
            'func': self.TR_span_func,
            'parents': ['sequence_start', 'TR', 'time_dim', 'frequency_dim', 'phase_dim', 'slice_dim', 'RF_dim', 'signal_dim']
        }
        
        node_specs['frequency_hover'] = {
            'func': self.frequency_hover_func,
            'parents': []
        }
        
        node_specs['phase_hover'] = {
            'func': self.phase_hover_func,
            'parents': []
        }
        
        node_specs['slice_hover'] = {
            'func': self.slice_hover_func,
            'parents': []
        }
        
        node_specs['RF_hover'] = {
            'func': self.RF_hover_func,
            'parents': []
        }
        
        node_specs['signal_hover'] = {
            'func': self.signal_hover_func,
            'parents': []
        }
        
        node_specs['frequency_board'] = {
            'func': self.frequency_board_func,
            'parents': ['time_dim', 'frequency_dim', 'frequency_objects', 'TR_span', 'frequency_hover']
        }

        node_specs['phase_board'] = {
            'func': self.phase_board_func,
            'parents': ['time_dim', 'phase_dim', 'phase_objects', 'TR_span', 'phase_hover']
        }

        node_specs['slice_board'] = {
            'func': self.slice_board_func,
            'parents': ['time_dim', 'slice_dim', 'slice_objects', 'TR_span', 'slice_hover']
        }

        node_specs['RF_board'] = {
            'func': self.RF_board_func,
            'parents': ['time_dim', 'RF_dim', 'RF_objects', 'TR_span', 'RF_hover']
        }

        node_specs['signal_board'] = {
            'func': self.signal_board_func,
            'parents': ['time_dim', 'signal_dim', 'signal_objects', 'ADC_objects', 'TR_span', 'signal_hover']
        }

        node_specs['sequence_plot'] = {
            'func': self.sequence_plot_func,
            'parents': ['frequency_board', 'phase_board', 'slice_board', 'RF_board', 'signal_board']
        }
        
        node_specs['kspace'] = {
            'func': self.kspace_func,
            'parents': ['kspace_type', 'show_processed_kspace', 'oversampled_recon_matrix', 'FOV', 'recon_matrix', 'full_k_matrix', 'zerofilled_kspace', 'kspace_exponent', 'gridded_kspace', 'k_grid_axes']
        }
        
        node_specs['zerofilled_kspace'] = {
            'func': self.zerofilled_kspace_func,
            'parents': ['apodized_kspace', 'oversampled_recon_matrix']
        }
        
        node_specs['image_array'] = {
            'func': self.image_array_func,
            'parents': ['oversampled_recon_matrix', 'full_k_matrix', 'recon_matrix', 'zerofilled_kspace']
        }
        
        node_specs['image'] = {
            'func': self.image_func,
            'parents': ['image_type', 'recon_matrix', 'FOV', 'image_array']
        }

        self.graph = build_graph(node_specs)

        self.set_reference_SNR()

        self.derived_params = ['FOV_bandwidth', 'FW_shift', 'SNR', 'name', 'num_shots', 'recon_voxel_F', 'recon_voxel_P', 'reference_SNR', 'relative_SNR', 'scantime', 'spoke_angle', 'voxel_F', 'voxel_P', 'shot_label']

    def init_bounds(self):
        self.param.object.objects = phantom.get_phantom_names()
        self.param.field_strength.objects=[1.5, 3.0]
        self.param.parameter_style.objects=['Matrix and Pixel BW', 'voxel_size and Fat/water shift', 'Matrix and FOV BW']
        self.param.frequency_direction.objects=constants.DIRECTIONS.keys()
        self.param.trajectory.objects=constants.TRAJECTORIES[:2]
        self.param.radial_factor.bounds=(0.1, 4.)
        self.param.radial_FOV_oversampling.bounds=(1, 2)
        self.param.sequence_type.objects=constants.SEQUENCES
        self.param.NSA.bounds=(1, 16)
        self.param.partial_Fourier.bounds=(.6, 1)
        self.param.turbo_factor.bounds=(1, 64)
        self.param.shot.bounds=(1, 1)
        self.param.image_type.objects=constants.OPERATORS.keys()
        self.param.kspace_type.objects=constants.OPERATORS.keys()
        self.param.kspace_exponent.bounds=(0.1, 1)
        self.param.apodization_alpha.bounds=(.01, 1)
        for par, values in param_values.items():
            self.param[par].objects=values

    def get_params(self):
        return {param: self.__getattribute__(param) for param in self.param.values().keys() if param not in self.derived_params}

    def set_params(self, settings):
        self.init_bounds()
        self.param.update(settings)

    def set_param_bounds(self, par, minval=None, maxval=None):
        if isinstance(minval, list):
            minval = max(minval) if minval else None
        if isinstance(maxval, list):
            maxval = min(maxval) if maxval else None
        curval = getattr(self, par.name)
        if type(par) is param.parameters.Selector:
            return self.set_param_discrete_bounds(par, curval, minval, maxval)
        if curval < minval:
            warnings.warn(f'trying to set {par.name} bounds above current value ({minval} > {curval})')
            minval = curval
            self.outbound_params.add(par.name)
        if curval > maxval:
            warnings.warn(f'trying to set {par.name} bounds below current value ({maxval} < {curval})')
            maxval = curval
            self.outbound_params.add(par.name)
        par.bounds = (minval, maxval)

    def conflict_solved(self, par):
        self.outbound_params.remove(par.name)
        if callable(getattr(par.objects, 'keys', None)) and 'outbound' in par.objects.keys():
            par.objects = dict(i for i in par.objects.items() if 'outbound' not in i)
        print(f'Param {par.name} no longer conflicting')

    def set_param_discrete_bounds(self, par, curval, minval=None, maxval=None):
        values = param_values[par.name]
        if minval is None:
            minval = min(values.values() if isinstance(values, dict) else values)
        if maxval is None:
            maxval = max(values.values() if isinstance(values, dict) else values)
        values = {k: v for k, v in values.items() if minval <= v <= maxval} if isinstance(values, dict) else [v for v in values if minval <= v <= maxval]
        
        outbound = (isinstance(values, list) and curval not in values) or (isinstance(values, dict) and curval not in values.values())
        if outbound:
            if (minval > maxval):
                warnings.warn(f'{par.name} has illegal bounds, minval > maxval ({minval} > {maxval})')
            else:
                warnings.warn(f'{par.name} current value {curval} is outside its new bounds [{minval}, {maxval}]')
            if isinstance(values, dict):
                values['outbound'] = curval
            elif isinstance(values, list) and not values:
                values = [curval]
            self.outbound_params.add(par.name)
        par.objects = values
        if not outbound and par.name in self.outbound_params:
            self.conflict_solved(par)

    def set_closest(self, par, value=None):
        # par.objects could be dict or param.ListProxy
        values = [v for k, v in par.objects.items() if k != 'outbound'] if callable(getattr(par.objects, 'items', None)) else par.objects
        if not values:
            warnings.warn(f'Could not set {par.name} since no allowed values')
            return
        if par.name in self.outbound_params:
            self.conflict_solved(par)
        if value is None:
            value = getattr(self, par.name)
        setattr(self, par.name, min(values, key=lambda x: abs(x-value)))

    def _watch_object(self):
        if hasattr(self, 'phantom') and self.phantom['name']==self.object:
            return
        self.phantom = phantom.load(self.object, self.min_voxel_size)
        min_FOV = self.phantom['support']
        if self.frequency_direction=='left-right':
            min_FOV = min_FOV.reverse()
        with param.parameterized.batch_call_watchers(self):
            self.FOV_F = max(self.FOV_F, min_FOV[0])
            self.FOV_P = max(self.FOV_P, min_FOV[1])

    def _watch_parameter_style(self):
        for par in [self.param.voxel_F, self.param.voxel_P, self.param.matrix_F, self.param.matrix_P, self.param.recon_voxel_F, self.param.recon_voxel_P, self.param.recon_matrix_F, self.param.recon_matrix_P, self.param.pixel_bandwidth, self.param.FOV_bandwidth, self.param.FW_shift]:
            par.precedence = -1
        if self.parameter_style == 'voxel_size and Fat/water shift':
            self.param.voxel_F.precedence = 4
            self.param.voxel_P.precedence = 4
            self.param.recon_voxel_F.precedence = 5
            self.param.recon_voxel_P.precedence = 5
            self.param.FW_shift.precedence = 2
            self._watch_recon_matrix_F()
            self._watch_recon_matrix_P()
            self._watch_pixel_bandwidth()
        else:
            self.param.matrix_F.precedence = 4
            self.param.matrix_P.precedence = 4
            self.param.recon_matrix_F.precedence = 5
            self.param.recon_matrix_P.precedence = 5
            if self.parameter_style == 'Matrix and Pixel BW':
                self.param.pixel_bandwidth.precedence = 2
            elif self.parameter_style == 'Matrix and FOV BW':
                self.param.FOV_bandwidth.precedence = 2
                self.update_matrix_F_bounds()

    def _watch_FOV_F(self):
        with param.parameterized.batch_call_watchers(self):
            if self.parameter_style=='voxel_size and Fat/water shift' or self.is_radial.value:
                self.set_closest(self.param.matrix_F, self.FOV_F/self.voxel_F)
                self.set_closest(self.param.recon_matrix_F, self.FOV_F/self.recon_voxel_F)
            self.update_FOV_bandwidth_objects()
            self.update_voxel_F_objects()
            self.update_recon_voxel_F_objects()
            self.set_closest(self.param.voxel_F, self.FOV_F/self.matrix_F)
            self.set_closest(self.param.recon_voxel_F, self.FOV_F/self.recon_matrix_F)
            self.param.trigger('voxel_F', 'recon_voxel_F')

    def _watch_FOV_P(self):
        with param.parameterized.batch_call_watchers(self):
            if self.parameter_style=='voxel_size and Fat/water shift' or self.is_radial.value:
                self.set_closest(self.param.matrix_P, self.FOV_P/self.voxel_P)
                self.set_closest(self.param.recon_matrix_P, self.FOV_P/self.recon_voxel_P)
            self.update_voxel_P_objects()
            self.update_recon_voxel_P_objects()
            self.set_closest(self.param.voxel_P, self.FOV_P/self.matrix_P)
            self.set_closest(self.param.recon_voxel_P, self.FOV_P/self.recon_matrix_P)
            self.param.trigger('voxel_P', 'recon_voxel_P')

    def _watch_phase_oversampling(self):
        self._watch_FOV_P()

    def _watch_radial_factor(self):
        pass

    def _watch_num_shots(self):
        pass

    def _watch_matrix_F(self):
        with param.parameterized.batch_call_watchers(self):
            self.update_FOV_bandwidth_objects()
            if self.parameter_style == 'Matrix and FOV BW':
                self.set_closest(self.param.pixel_bandwidth, FOV_BW_to_pixel_BW(self.FOV_bandwidth, self.matrix_F))
            else:
                self.set_closest(self.param.FOV_bandwidth, pixel_BW_to_FOV_BW(self.pixel_bandwidth, self.matrix_F))
                self.param.trigger('FOV_bandwidth')
            self.update_voxel_F_objects()
            self.set_closest(self.param.voxel_F, self.FOV_F / self.matrix_F)
            self.set_param_bounds(self.param['recon_matrix_F'], minval=self.matrix_F)
            self.update_recon_voxel_F_objects()
            self.set_closest(self.param.recon_matrix_F, self.matrix_F * self.rec_acq_ratio_F)
            self.param.trigger('voxel_F', 'recon_voxel_F')
            if self.is_radial.value:
                self.set_closest(self.param.matrix_P, self.matrix_F * self.FOV_P / self.FOV_F)

    def _watch_matrix_P(self):
        with param.parameterized.batch_call_watchers(self):
            self.update_voxel_P_objects()
            self.set_closest(self.param.voxel_P, self.FOV_P / self.matrix_P)
            self.set_param_bounds(self.param['recon_matrix_P'], minval=self.matrix_P)
            self.update_recon_voxel_P_objects()
            self.set_closest(self.param.recon_matrix_P, self.matrix_P * self.rec_acq_ratio_P)
            self.param.trigger('voxel_P', 'recon_voxel_P')
            if self.is_radial.value:
                self.set_closest(self.param.matrix_F, self.matrix_P * self.FOV_F / self.FOV_P)

    def _watch_voxel_F(self):
        if self.param.voxel_F.precedence > 0:
            self.set_closest(self.param.matrix_F, self.FOV_F / self.voxel_F)

    def _watch_voxel_P(self):
        if self.param.voxel_P.precedence > 0:
            self.set_closest(self.param.matrix_P, self.FOV_P / self.voxel_P)

    def _watch_recon_voxel_F(self):
        if self.param.recon_voxel_F.precedence > 0:
            self.set_closest(self.param.recon_matrix_F, self.FOV_F / self.recon_voxel_F)

    def _watch_recon_voxel_P(self):
        if self.param.recon_voxel_P.precedence > 0:
            self.set_closest(self.param.recon_matrix_P, self.FOV_P / self.recon_voxel_P)

    def _watch_slice_thickness(self):
        pass

    def _watch_radial_FOV_oversampling(self):
        pass

    def _watch_frequency_direction(self):
        for p in [self.param.FOV_F, self.param.FOV_P, self.param.matrix_F, self.param.matrix_P, self.param.recon_matrix_F, self.param.recon_matrix_P]:
            if ' x' in p.label:
                p.label = p.label.replace(' x', ' y')
            elif ' y' in p.label:
                p.label = p.label.replace(' y', ' x')

    def _watch_trajectory(self):
        if self.is_radial.value:
            self.partial_Fourier = 1.
            self.param.partial_Fourier.precedence = -5
            self.param.frequency_direction.precedence = -1
            self.phase_oversampling = 0.
            self.param.phase_oversampling.precedence = -3
            self.param.radial_factor.precedence = 3
            # set isotropic voxel_size:
            if (self.FOV_F / self.matrix_F < self.FOV_P / self.matrix_P):
                self.set_closest(self.param.matrix_P, self.matrix_F * self.FOV_P / self.FOV_F)
            else:
                self.set_closest(self.param.matrix_F, self.matrix_P * self.FOV_F / self.FOV_P)
        else:
            self.param.partial_Fourier.precedence = 5
            self.param.frequency_direction.precedence = 1
            self.param.phase_oversampling.precedence = 3
            self.param.radial_factor.precedence = -3
        self.update_labels_by_trajectory()

    def _watch_field_strength(self):
        with param.parameterized.batch_call_watchers(self):
            self.update_FW_shift_objects()
            self.set_closest(self.param.FW_shift, pixel_BW_to_shift(self.pixel_bandwidth, self.field_strength))
            self.param.trigger('FW_shift')
            self._watch_FatSat() # since fatsat pulse duration depends on field_strength

    def _watch_pixel_bandwidth(self):
        with param.parameterized.batch_call_watchers(self):
            self.set_closest(self.param.FW_shift, pixel_BW_to_shift(self.pixel_bandwidth, self.field_strength))
            self.set_closest(self.param.FOV_bandwidth, pixel_BW_to_FOV_BW(self.pixel_bandwidth, self.matrix_F))

    def _watch_FW_shift(self):
        if self.param.FW_shift.precedence > 0:
            self.set_closest(self.param.pixel_bandwidth, shift_to_pixel_BW(self.FW_shift, self.field_strength))

    def _watch_FOV_bandwidth(self):
        if self.param.FOV_bandwidth.precedence > 0:
            self.set_closest(self.param.pixel_bandwidth, FOV_BW_to_pixel_BW(self.FOV_bandwidth, self.matrix_F))

    def _watch_NSA(self):
        pass

    def _watch_partial_Fourier(self):
        pass

    def _watch_turbo_factor(self):
        self.update_labels_by_trajectory()
        self.update_EPI_factor_objects()

    def _watch_EPI_factor(self):
        self.update_labels_by_trajectory()
        self.update_turbo_factor_bounds()

    def _watch_shot(self):
        pass

    def _watch_sequence(self):
        self.param.FA.precedence = 1 if self.sequence=='Spoiled Gradient Echo' else -1
        self.param.TI.precedence = 1 if self.sequence=='Inversion Recovery' else -1
        if self.sequence=='Spoiled Gradient Echo':
            self.turbo_factor = 1
            self.param.turbo_factor.precedence = -6
        else:
            self.param.turbo_factor.precedence = 6

    def _watch_TE(self):
        pass

    def _watch_TR(self):
        pass

    def _watch_TI(self):
        pass

    def _watch_FA(self):
        pass

    def _watch_FatSat(self):
        pass

    def _watch_homodyne(self):
        pass

    def _watch_do_apodize(self):
        if self.do_apodize:
            self.param.apodization_alpha.precedence = abs(self.param.apodization_alpha.precedence)
        else:
            self.param.apodization_alpha.precedence = -abs(self.param.apodization_alpha.precedence)

    def _watch_apodization_alpha(self):
        pass

    def _watch_do_zerofill(self):
        pass

    def _watch_recon_matrix_F(self):
        self.rec_acq_ratio_F = self.recon_matrix_F / self.matrix_F
        self.set_closest(self.param.recon_voxel_F, self.FOV_F / self.recon_matrix_F)

    def _watch_recon_matrix_P(self):
        self.rec_acq_ratio_P = self.recon_matrix_P / self.matrix_P
        self.set_closest(self.param.recon_voxel_P, self.FOV_P / self.recon_matrix_P)

    def _watch_reference_tissue(self):
        pass

    def resolve_conflicts(self, max_TR=False):
        if max_TR:
            self.outbound_params.add('TR')
        if self.outbound_params:
            for par in ['TR', 'TI', 'TE', 'pixel_bandwidth']:
                if par in self.outbound_params:
                    value = getattr(self, par)
                    if par=='TR' and max_TR:
                        warnings.warn('Resolving conflict by maximizing TR')
                        self.TR = list(param_values['TR'].values())[-1]
                    else:
                        warnings.warn(f'Resolving conflict: {par}')
                    self.set_closest(self.param[par], value) # Set back param within bounds
        if self.outbound_params:
            if not max_TR:
                self.resolve_conflicts(max_TR=True)
            else:
                warnings.warn(f'Unresolved conflict: {self.outbound_params}')
    
    def set_min_TR(self, min_TR):
        self.set_param_bounds(self.param.TR, minval=min_TR)

    def set_TE_bounds(self, TR, min_TR, TE, min_TE):
        #TODO: shouldn't depend on TE!
        max_TE = TR - min_TR + TE
        self.set_param_bounds(self.param.TE, minval=min_TE, maxval=max_TE)

    def set_max_TI(self, sequence_type, TR, min_TR, TI):
        if sequence_type != 'Inversion Recovery':
            return
        max_TI = TR - min_TR + TI
        self.set_param_bounds(self.param.TI, maxval=max_TI)

    def set_labels_by_trajectory(self, shot_label):
        self.param.shot.label = f'Displayed {shot_label}'
        self.param.radial_factor.label = f'{shot_label.capitalize()} sampling factor'

    def set_trajectory_objects(self, EPI_factor, turbo_factor):
        # Label radial trajectory 'Radial' or 'PROPELLER' depending on nLines per shot
        self.param.trajectory.objects = constants.TRAJECTORIES
        invalid, updated = ('PROPELLER', 'Radial') if (EPI_factor * turbo_factor == 1) else ('Radial', 'PROPELLER')
        if self.trajectory == invalid:
            self.trajectory = updated
        self.param.trajectory.objects = [t for t in constants.TRAJECTORIES if t != invalid]

    def set_BW_bounds(self, matrix_F, FOV_F, is_gradient_echo, gre_max_read_duration, RF_refocusing, turbo_factor, EPI_factor, TE, RF_excitation, readtrain_spacing, phaser_duration, max_blip_dur, slice_select_refocusing, TR, sequence_start, spoiler):
        # See paramBounds.tex for formulae relating to the readout board
        s = self.max_slew
        A = 1e3 * matrix_F / (FOV_F * constants.GYRO) # readout area
        # min limit imposed by maximum gradient amplitude:
        min_read_durations = [A / self.max_amp]
        max_read_durations = []
        if is_gradient_echo:
            max_read_durations.append(gre_max_read_duration)
        else: # spin echo
            refocusing_dur = RF_refocusing[0]['dur_f']
            if turbo_factor==1 and EPI_factor==1: # prephaser should only be limiting for pure spin echo
                # min limit imposed by prephaser duration tp:
                tp = TE/2 - refocusing_dur/2 - RF_excitation['dur_f']/2
                h = s * tp / 2
                h = min(h, self.max_amp)
                denom = 2*h*s*tp - s*A - 2*h**2
                min_read_durations.append(np.sqrt(A**2/denom) if denom > 0 else np.inf)
            idle_space = readtrain_spacing - RF_refocusing[0]['dur_f']
            # max limit imposed by phaser:
            max_read_durations.append((idle_space - 2 * phaser_duration - max_blip_dur * (EPI_factor-1))/EPI_factor)
            # tr is half the maximum readout gradient duration
            tr = ((idle_space) / EPI_factor) / 2
            # max limit imposed by readout rise time:
            radicand = tr**2 - 2*A/s
            if radicand >= 0:
                max_read_durations.append(tr + np.sqrt(radicand))
            else:
                max_read_durations.append(0)
            # max limit imposed by slice select refocusing down ramp time:
            max_read_durations.append((tr - slice_select_refocusing[0]['risetime_f']) * 2)
            # readtrain_spacing may be limited by TR:
            read_end_by_TR = (TR - (-sequence_start) - spoiler['dur_f'])
            read_end_by_else = readtrain_spacing * (turbo_factor + 1/2) - refocusing_dur/2
            max_read_durations.append((tr - (read_end_by_else-read_end_by_TR)) * 2)
        min_read_duration, max_read_duration = max(min_read_durations), min(max_read_durations)
        small = 1e-2 # to avoid roundoff errors
        min_pixel_BW = 1e3 / max_read_duration + small if max_read_duration > 0 else np.inf
        max_pixel_BW = 1e3 / min_read_duration - small if min_read_duration > 0 else np.inf
        self.set_param_bounds(self.param.pixel_bandwidth, minval=min_pixel_BW, maxval=max_pixel_BW)

    def set_matrix_F_bounds(self, max_readout_area, FOV_F, parameter_style, FOV_bandwidth):
        min_matrix_F = []
        max_matrix_F = [max_readout_area * 1e-3 * FOV_F * constants.GYRO]
        if parameter_style == 'Matrix and FOV BW':
            # TODO: this could be solved better
            min_matrix_F.append(FOV_bandwidth / list(self.param.pixel_bandwidth.objects.values())[-1])
            max_matrix_F.append(FOV_bandwidth / list(self.param.pixel_bandwidth.objects.values())[0])
        self.set_param_bounds(self.param.matrix_F, minval=min_matrix_F, maxval=max_matrix_F)

    def set_matrix_P_bounds(self, max_phaser_area, FOV_P):
        max_matrix_P = int(max_phaser_area * 2e-3 * FOV_P * constants.GYRO) + 1
        self.set_param_bounds(self.param.matrix_P, maxval=max_matrix_P)

    def set_FOV_F_bounds(self, matrix_F, max_readout_area, parameter_style, voxel_F, recon_voxel_F):
        min_FOV = [1e3 * matrix_F / (max_readout_area * constants.GYRO) if max_readout_area > 0 else np.inf]
        max_FOV = []
        if parameter_style == 'voxel_size and Fat/water shift':
            # TODO: this could be solved better
            min_FOV.append(voxel_F * self.param.matrix_F.objects[0])
            min_FOV.append(recon_voxel_F * self.param.recon_matrix_F.objects[0])
            max_FOV.append(voxel_F * self.param.matrix_F.objects[-1])
            max_FOV.append(recon_voxel_F * self.param.recon_matrix_F.objects[-1])
        self.set_param_bounds(self.param.FOV_F, minval=min_FOV, maxval=max_FOV)

    def set_FOV_P_bounds(self, matrix_P, max_phaser_area, parameter_style, voxel_P, recon_voxel_P):
        min_FOV = [(matrix_P - 1) / (max_phaser_area * constants.GYRO * 2e-3)]
        max_FOV = []
        if parameter_style == 'voxel_size and Fat/water shift':
            # TODO: this could be solved better
            min_FOV.append(voxel_P * self.param.matrix_P.objects[0])
            min_FOV.append(recon_voxel_P * self.param.recon_matrix_P.objects[0])
            max_FOV.append(voxel_P * self.param.matrix_P.objects[-1])
            max_FOV.append(recon_voxel_P * self.param.recon_matrix_P.objects[-1])
        self.set_param_bounds(self.param.FOV_P, minval=min_FOV, maxval=max_FOV)

    def set_FW_shift_objects(self, field_strength):
        self.param.FW_shift.objects = {f'{format_float(shift, 2)} pixels': shift for shift in [pixel_BW_to_shift(pBW, field_strength) for pBW in list(self.param.pixel_bandwidth.objects.values())[::-1]]}

    def set_FOV_bandwidth_objects(self, matrix_F):
        self.param.FOV_bandwidth.objects = {f'±{format_float(bw, 3)} kHz': bw for bw in [pixel_BW_to_FOV_BW(pBW, matrix_F) for pBW in self.param.pixel_bandwidth.objects.values()]}

    def set_voxel_F_objects(self, FOV_F):
        self.param.voxel_F.objects = {f'{format_float(voxel, 3)} mm': voxel for voxel in [FOV_F / matrix for matrix in self.param.matrix_F.objects[::-1]]}

    def set_voxel_P_objects(self, FOV_P):
        self.param.voxel_P.objects = {f'{format_float(voxel, 3)} mm': voxel for voxel in [FOV_P / matrix for matrix in self.param.matrix_P.objects[::-1]]}

    def set_recon_voxel_F_objects(self, FOV_F):
        self.param.recon_voxel_F.objects = {f'{format_float(voxel, 3)} mm': voxel for voxel in [FOV_F / matrix for matrix in self.param.recon_matrix_F.objects[::-1]]}

    def set_recon_voxel_P_objects(self, FOV_P):
        self.param.recon_voxel_P.objects = {f'{format_float(voxel, 3)} mm': voxel for voxel in [FOV_P / matrix for matrix in self.param.recon_matrix_P.objects[::-1]]}

    def set_slice_thickness_bounds(self, RF_excitation, is_gradient_echo, RF_refocusing, sequence_type, RF_inversion, TR, spoiler, sampling_windows):
        min_thks = [RF_excitation['FWHM_f'] / (self.max_amp * constants.GYRO)]
        if not is_gradient_echo:
            min_thks.append(RF_refocusing[0]['FWHM_f'] / (self.max_amp * constants.GYRO))
        if sequence_type == 'Inversion Recovery':
            min_thks.append(RF_inversion['FWHM_f'] / (self.max_amp * constants.GYRO) * self.inversion_thk_factor)
        
        # Constraint due to TR: 
        if sequence_type == 'Inversion Recovery':
            max_risetime = TR - (spoiler['time'][-1] - RF_inversion['time'][0])
            max_amp = self.max_slew * max_risetime
            min_thks.append(RF_inversion['FWHM_f'] / (max_amp * constants.GYRO))
        else:
            max_risetime = TR - (spoiler['time'][-1] - RF_excitation['time'][0])
            max_amp = self.max_slew * max_risetime
            min_thks.append(RF_excitation['FWHM_f'] / (max_amp * constants.GYRO))
        
        # See paramBounds.tex for formulae
        s = self.max_slew
        d = RF_excitation['dur_f']
        if is_gradient_echo: # Constraint due to slice rephaser
            t = sampling_windows[0][0]['time'][0]
            h = s * (t - np.sqrt(t**2/2 + d**2/8))
            h = min(h, self.max_amp)
            A = d * (np.sqrt((d*s+2*h)**2 - 8*h*(h-s*(t-d/2))) - d*s - 2*h) / 2
        else: # Spin echo: Constraint due to slice rephaser and refocusing slice select rampup
            t = RF_refocusing[0]['time'][0]
            h = s * (np.sqrt(2*(d + 2*t)**2 - 4*d**2) - d - 2*t) / 4
            h = min(h, self.max_amp)
            A = (np.sqrt((d*(d*s + 4*h))**2 - 4*d**2*h*(d*s + 2*h - 2*s*t)) - d*(d*s + 4*h)) / 2
        Be = RF_excitation['FWHM_f']
        min_thks.append(Be * d / (constants.GYRO * A)) # mm
        
        self.set_param_bounds(self.param.slice_thickness, minval=min_thks)

    def set_turbo_factor_bounds(self, matrix, phase_dir, partial_Fourier, EPI_factor):
        # turbo_factor must equal 1 when the EPI_factor is even
        if not self.EPI_factor%2:
            self.param.turbo_factor.bounds = (1, 1)
            self.param.turbo_factor.constant = True
            return
        max_turbo_factor = int(np.floor(matrix[phase_dir] * partial_Fourier / EPI_factor * 2)) # let's limit phase oversampling to 2
        max_turbo_factor = min(max_turbo_factor, 64)
        self.param.turbo_factor.bounds = (1, max_turbo_factor)
        self.param.turbo_factor.constant = False

    def set_EPI_factor_objects(self, matrix, phase_dir, partial_Fourier, turbo_factor):
        max_EPI_factor = int(np.floor(matrix[phase_dir] * partial_Fourier / turbo_factor * 2)) # let's limit phase oversampling to 2
        self.set_param_bounds(self.param.EPI_factor, maxval=max_EPI_factor)
        # EPI_factor must be odd for turbo spin echo (GRASE)
        if self.turbo_factor > 1:
            self.param.EPI_factor.objects = [v for v in self.param.EPI_factor.objects if v%2]
    
    def set_homodyne_visibility(self, num_blank_lines, is_radial):
        self.param.homodyne.precedence = -1 if (num_blank_lines == 0 or is_radial) else 1

    def set_reference_tissue_objects(self, tissues):
        self.param.reference_tissue.objects = tissues
        self.reference_tissue = tissues[0]
    
    def set_shot_bounds(self, num_shots):
        self.param.shot.bounds = (1, num_shots)
        self.shot = min(self.shot, num_shots)
    
    def min_TE_func(self, EPI_factor, centermost_rf_echo, centermost_gr_echo, min_readtrain_spacing, gr_echo_spacing):
        if EPI_factor == 1: # flexible segment order for (turbo) spin echo
            min_centermost_rf_echo = 0
            min_centermost_gr_echo = 0
        else: # linear segment order for EPI and GRASE
            # TODO: evaluate forward/reverse linear order
            min_centermost_rf_echo = centermost_rf_echo
            min_centermost_gr_echo = centermost_gr_echo
        return TE_from_centermost_echoes(min_readtrain_spacing, min_centermost_rf_echo, gr_echo_spacing, min_centermost_gr_echo, EPI_factor)

    def min_TR_func(self, spoiler, sequence_start):
        return spoiler['time'][-1] - sequence_start
    
    def gre_read_start_to_kcenter_func(self, is_gradient_echo, phasers, rephasers, TE, RF_excitation, read_prephaser, readout_risetime, slice_select_excitation, slice_select_rephaser):
        if not is_gradient_echo:
            return None
        min_phaser_time = min([grads[0]['dur_f'] for grads in [phasers, rephasers]])
        read_start_to_kcenter = TE - RF_excitation['dur_f']/2
        read_start_to_kcenter -= max(
            read_prephaser['dur_f'] + readout_risetime, # TODO: consider maximum dur+risetime, not only current (difficult!)
            min_phaser_time,
            slice_select_excitation['risetime_f'] + slice_select_rephaser['dur_f'])
        return read_start_to_kcenter
    
    def gre_kcenter_to_read_end_func(self, is_gradient_echo, TR, sequence_start, spoiler, TE):
        if not is_gradient_echo:
            return None
        return (TR - (-sequence_start) - spoiler['dur_f']) - TE
    
    def gre_max_read_duration_func(self, gre_read_start_to_kcenter, gre_kcenter_to_read_end, centermost_gr_echo, readout_risetime, EPI_factor):
        if gre_read_start_to_kcenter is None:
            return None
        # simplification: use current risetime (self.readout_risetime)
        num_early_readouts = centermost_gr_echo + 1/2
        num_early_ramps = centermost_gr_echo * 2
        # max limit imposed by TE:
        max_read_dur_early = ((gre_read_start_to_kcenter - num_early_ramps * readout_risetime) / num_early_readouts)
        num_late_readouts = EPI_factor - num_early_readouts
        num_late_ramps = (EPI_factor - 1) * 2 - num_early_ramps
        # max limit imposed by TR:
        max_read_dur_late = ((gre_kcenter_to_read_end - num_late_ramps * readout_risetime) / num_late_readouts)
        return min(max_read_dur_early, max_read_dur_late)
    
    def max_readout_area_func(self, pixel_bandwidth, is_gradient_echo, centermost_gr_echo, RF_excitation, phaser_duration, slice_select_excitation, slice_select_rephaser, max_blip_dur, readtrain_spacing, RF_refocusing, EPI_factor):
        max_readout_areas = []
        # See paramBounds.tex for formulae
        d = 1e3 / pixel_bandwidth # readout duration
        s = self.max_slew
        if is_gradient_echo:
            N = centermost_gr_echo + 1/2
            M = centermost_gr_echo * 2 + 1
            t = self.TE - RF_excitation['dur_f']/2
            v = 0 # gap between readouts
            for _ in range(2): # update readout gap after first pass
                if (M > 1):
                    # max wrt G slice or G phase:
                    q = t - max(phaser_duration,
                                slice_select_excitation['risetime_f'] + slice_select_rephaser['dur_f'])
                    A = d*s*(q - N*(d+v) + v/2) / (M-1) # eq. 15
                    max_readout_areas.append(A)
                # max wrt G read:
                h_roots = np.roots([8*(3-2*M), 4*s*(t*(2*M-6)+d*(2*N-M)+v*(2*N-1)), s**2*(4*t**2-d**2)]) # eq. 12
                h = min([h for h in h_roots if h>0] + [self.max_amp]) # truncate prephaser amp to max amp
                A_roots = np.roots([1, d*(d*s + 2*M*h), d**2*h*(2*h-s*(2*t-2*N*(d+v)+v))]) # eq. 13
                if np.all(A_roots<0):
                    return 0 # no positive roots
                A = min([A for A in A_roots if A>0])
                max_readout_areas.append(A)
                read_risetime = min(max_readout_areas) / (d * s)
                v = max(max_blip_dur - 2 * read_risetime, 0)
        else: # (turbo) spin echo / GRASE
            # limit by half readout duration tr:
            tr = (readtrain_spacing - RF_refocusing[0]['dur_f']) / EPI_factor / 2
            Ar = d*s* tr - d**2*s/2
            max_readout_areas.append(Ar)
            # limit by prephaser duration tp:
            tp = (readtrain_spacing - RF_refocusing[0]['dur_f'] - RF_excitation['dur_f'])/2
            h = s * tp / 2
            h = min(h, self.max_amp)
            Ap = d * (np.sqrt((d*s)**2 - 8*h*(h-s*tp)) - d*s) / 2
            max_readout_areas.append(Ap)
        max_readout_areas.append(self.max_amp * 1e3 / pixel_bandwidth) # max wrt max_amp
        return min(max_readout_areas)

    def max_phaser_area_func(self, is_gradient_echo, readtrain_spacing, RF_excitation, gre_echo_train_dur, readout_risetime, RF_refocusing):
        if is_gradient_echo:
            max_phaser_duration = readtrain_spacing - RF_excitation['dur_f']/2 - gre_echo_train_dur/2 + readout_risetime
        else:
            max_phaser_duration = (readtrain_spacing - RF_refocusing[0]['dur_f'] - gre_echo_train_dur)/2 + readout_risetime
        max_risetime = self.max_amp / self.max_slew
        if max_phaser_duration > 2 * max_risetime: # trapezoid maxPhaser
            max_phaserarea = (max_phaser_duration - max_risetime) * self.max_amp
        else: # triangular maxPhaser
            max_phaserarea = (max_phaser_duration/2)**2 * self.max_slew
        return max_phaserarea
    
    def min_readtrain_spacing_func(self, is_gradient_echo, RF_excitation, gre_echo_train_dur, readout_risetime, read_prephaser, phaser_duration, slice_select_excitation, slice_select_rephaser, RF_refocusing, slice_select_refocusing):
        # Get shortest spacing for (center of) gradient echo trains
        # Equals center position of gradient echo (train) for gradient echo sequences
        # Equals rf echo spacing for spin echo sequences
        if is_gradient_echo:
            spacing = (RF_excitation['dur_f'] + gre_echo_train_dur) / 2 - readout_risetime
            spacing += max(
                read_prephaser['dur_f'] + readout_risetime,
                phaser_duration,
                slice_select_excitation['risetime_f'] + slice_select_rephaser['dur_f']
            )
        else: # spin echo
            # before refocusing pulse:
            left_side = (RF_excitation['dur_f'] + RF_refocusing[0]['dur_f']) / 2
            left_side += max(
                read_prephaser['dur_f'], 
                slice_select_excitation['risetime_f'] + slice_select_rephaser['dur_f'] + (slice_select_refocusing[0]['risetime_f'])
            )
            # after refocusing pulse:
            right_side = (RF_refocusing[0]['dur_f'] + gre_echo_train_dur) / 2 - readout_risetime
            right_side += max(
                readout_risetime,
                phaser_duration,
                slice_select_refocusing[0]['risetime_f']
            )
            spacing = max(left_side, right_side) * 2
        return spacing
    
    def centermost_gr_echo_func(self, central_segment, reverse_linear_order, num_segm, turbo_factor):
        # get index of gradient echo closest to k-space center
        #TODO: opposite gre order or different rounding (ceil/floor) may minimize TE
        centermost_gr_echo = central_segment // turbo_factor
        return centermost_gr_echo

    def centermost_rf_echo_func(self, is_gradient_echo, EPI_factor, central_segment, turbo_factor, TE, min_readtrain_spacing):
        # get index if RF echo closest to k-space center
        if is_gradient_echo:
            return 0
        if EPI_factor > 1: # linear segment order for EPI and GRASE
            #TODO: opposite rfe order or different rounding (ceil/floor) may minimize TE
            return central_segment % turbo_factor
        return min(int(np.floor(TE / min_readtrain_spacing)) - 1, turbo_factor - 1)
    
    def readtrain_spacing_func(self, EPI_factor, gr_echo_spacing, TE, centermost_gr_echo, centermost_rf_echo):
        # Equals center position of gradient echo (train) for gradient echo sequences
        # Equals rf echo spacing for spin echo sequences
        spin_echo_time = TE
        if EPI_factor > 1:
            spin_echo_time -= readtrain_shift(gr_echo_spacing, centermost_gr_echo, EPI_factor)
        return spin_echo_time / (centermost_rf_echo + 1)
        
    def k_read_axis_func(self, freq_dir, FOV, matrix, is_radial, fantom, radial_FOV_oversampling):
        voxel_size = FOV[freq_dir] / matrix[freq_dir]
        if not is_radial:
            num_samples = matrix[freq_dir]
            # at least Nyquist sampling wrt phantom if loaded
            if FOV[freq_dir] < fantom['support'][freq_dir]:
                num_samples = int(np.ceil(fantom['support'][freq_dir] / voxel_size))
        else:
            maxFOV = max(max(fantom['support']), max(FOV))
            num_samples = int(np.ceil(maxFOV / voxel_size * radial_FOV_oversampling))
        return recon.get_k_axis(num_samples, voxel_size)

    def k_phase_axis_func(self, is_radial, num_measured_lines, matrix, phase_dir, phase_oversampling, FOV):
        if not is_radial:
            # oversampling may be higher than prescribed since num_shots must be integer:
            num_lines = max(num_measured_lines, int(np.ceil(matrix[phase_dir] * (1 + phase_oversampling / 100))))
            voxel_size = FOV[phase_dir] / matrix[phase_dir]
        else:
            num_lines = num_measured_lines # future: take undersampling into account
            voxel_size = max(FOV) / num_lines # corresponding to blade width
        return recon.get_k_axis(num_lines, voxel_size)

    def lines_to_measure_func(self, k_phase_axis, num_measured_lines):
        lines_to_measure = np.ones(len(k_phase_axis), dtype=bool)
        # undersample by partial Fourier:
        lines_to_measure[num_measured_lines:] = False
        assert(sum(lines_to_measure) == num_measured_lines)
        return lines_to_measure

    def num_segm_func(self, num_measured_lines, num_blades, num_shots):
        # number of k-space segments
        return int(num_measured_lines * num_blades / num_shots)

    def num_sym_lines_func(self, num_measured_lines, k_phase_axis):
        return 2 * num_measured_lines - len(k_phase_axis)

    def split_center_func(self, num_sym_lines, num_shots):
        # does center of k-space lie between two segments?
        return (num_sym_lines % num_shots == 0) and ((num_sym_lines / num_shots) % 2 == 0)

    def num_sym_segm_func(self, split_center, num_sym_lines, num_blades, num_shots):
        # number of k-space segments symmetric about center:
        if split_center:
            return int(num_sym_lines * num_blades / num_shots)
        return int(np.round((num_sym_lines * num_blades / num_shots - 1) / 2)) * 2 + 1

    def central_segment_func(self, num_segm, num_sym_segm):
        return num_segm - np.ceil(num_sym_segm / 2)
        
    def pe_table_func(self, EPI_factor, turbo_factor, num_sym_segm, centermost_rf_echo, is_radial, num_shots, reverse_linear_order, lines_to_measure):
        if EPI_factor == 1: # (turbo) spin echo
            segment_order = get_segment_order(turbo_factor, num_sym_segm, centermost_rf_echo)
            if is_radial:
                pe_table = [[[segment] for segment in segment_order] for shot in range(num_shots)]
            else:
                pe_table = [[[segment * num_shots + shot] for segment in segment_order] for shot in range(num_shots)]
        else: # EPI and GRASE
            order = -1 if reverse_linear_order else 1
            if is_radial:
                pe_table = [[list(range(rf_echo, sum(lines_to_measure), turbo_factor))[::order] for rf_echo in range(turbo_factor)][::order] for shot in range(num_shots)]
            else:
                pe_table = [[list(range(rf_echo * num_shots + shot, sum(lines_to_measure), num_shots * turbo_factor))[::order] for rf_echo in range(turbo_factor)][::order] for shot in range(num_shots)]
        return np.array(pe_table)

    def k_axes_func(self, freq_dir, phase_dir, k_read_axis, k_phase_axis, lines_to_measure):
        k_axes = [None]*2
        k_axes[freq_dir] = k_read_axis
        k_axes[phase_dir] = k_phase_axis[lines_to_measure]
        return k_axes
    
    def k_samples_func(self, k_axes, k_angles):
        k_samples = np.array(np.meshgrid(k_axes[0], k_axes[1])).T
        # rotate samples for each angle:
        rotmat = np.array([[np.cos(k_angles), -np.sin(k_angles)], 
                           [np.sin(k_angles),  np.cos(k_angles)]])
        return np.einsum('ijk,klm->ijml', k_samples, rotmat) # shape=(Nx, Ny, Nangles, 2)
    
    def k_grid_axes_func(self, is_radial, k_axes, FOV, matrix, fantom):
        if not is_radial:
            return k_axes.copy()
        k_grid_axes = [None, None]
        for dim in range(2):
            voxel_size = FOV[dim] / matrix[dim]
            matrix_dim = int(np.ceil(max(FOV[dim], fantom['support'][dim]) / voxel_size))
            k_grid_axes[dim] = recon.get_k_axis(matrix_dim, voxel_size)
        return k_grid_axes
    
    def plain_kspace_comps_func(self, is_radial, fantom, k_grid_axes, k_samples):
        if not is_radial:
            return recon.resample_kspace_Cartesian(fantom, k_grid_axes, shape=k_samples.shape[:-1])
        return recon.resample_kspace(fantom, k_samples)
        
    def thick_kspace_comps_func(self, slice_thickness, k_samples, plain_kspace_comps):
        # Lorenzian line shape to mimic slice thickness
        blur_factor = .5
        slice_thickness_filter = slice_thickness * np.exp(-blur_factor * slice_thickness * np.sqrt(np.sum(k_samples**2, axis=-1)))
        thick_kspace_comps = {}
        for tissue in plain_kspace_comps:
            thick_kspace_comps[tissue] = plain_kspace_comps[tissue] * slice_thickness_filter
        return thick_kspace_comps
        
    def signal_level_func(self, k_read_axis, lines_to_measure, num_blades, slice_thickness, FOV, matrix):
        return np.sqrt(len(k_read_axis) * sum(lines_to_measure) * num_blades) * slice_thickness * np.prod(FOV) / np.prod(matrix)

    def sampling_time_func(self, pixel_bandwidth, k_read_axis):
        # time of sample along (positive) readout relative to (k-space) center
        half_read_duration = .5e3 / pixel_bandwidth # msec
        return np.linspace(-half_read_duration, half_read_duration, len(k_read_axis))
        
    def noise_std_func(self, sampling_time, noise_gain, NSA, field_strength):
        dwell_time = np.diff(sampling_time[:2])[0]
        return noise_gain / np.sqrt(dwell_time * NSA) / field_strength

    def spin_echoes_func(self, lines_to_measure, pe_table, readtrain_spacing):
        spin_echoes = np.zeros((sum(lines_to_measure)))
        for ky in range(sum(lines_to_measure)):
            shot, rf_echo, gr_echo = np.argwhere(pe_table==ky)[0]
            spin_echoes[ky] = (rf_echo + 1) * readtrain_spacing
        return spin_echoes
    
    def time_after_excitation_func(self, lines_to_measure, pe_table, readouts, sampling_time, freq_dir, phase_dir):
        TEs = np.zeros((sum(lines_to_measure)))
        reverse = np.zeros((sum(lines_to_measure)), dtype=bool)
        for ky in range(sum(lines_to_measure)):
            shot, rf_echo, gr_echo = np.argwhere(pe_table==ky)[0]
            TEs[ky] = readouts[rf_echo][gr_echo]['center_f']
            reverse[ky] = readouts[rf_echo][gr_echo]['area_f'] < 0
        sampling_offset = np.expand_dims(sampling_time, axis=[dim for dim in range(3) if dim != freq_dir])
        time_after_excitation = np.expand_dims(TEs, axis=[dim for dim in range(3) if dim != phase_dir]) + sampling_offset
        # EPI rowflip:
        reverse_time_after_excitation = np.flip(time_after_excitation, axis=freq_dir)
        reverse = np.expand_dims(reverse, axis=[dim for dim in range(3) if dim != phase_dir])
        reverse = reverse.repeat(len(sampling_time), axis=freq_dir)
        time_after_excitation[reverse] = reverse_time_after_excitation[reverse]
        return time_after_excitation
        
    def time_relative_inphase_func(self, time_after_excitation, is_gradient_echo, spin_echoes, phase_dir):
        time_relative_inphase = time_after_excitation.copy()
        if not is_gradient_echo:
            # for spinecho, subtract Hahn echo position from time_after_excitation
            time_relative_inphase -= np.expand_dims(spin_echoes, axis=[dim for dim in range(3) if dim != phase_dir])
        return time_relative_inphase
        
    def dephasing_func(self, field_strength, time_relative_inphase):
        dephasing = {}
        for component, resonance in constants.FAT_RESONANCES.items():
            dephasing[component] = np.exp(2j*np.pi * constants.GYRO * field_strength * resonance['shift'] * time_relative_inphase * 1e-3)
        return dephasing

    def T2w_func(self, tissues, time_after_excitation, time_relative_inphase, field_strength):
        T2w = {}
        for tissue in tissues:
            T2w[tissue] = get_T2w(tissue, time_after_excitation, time_relative_inphase, field_strength)
            if constants.TISSUES[tissue]['FF'] > 0: # fat containing tissues
                for component in constants.FAT_RESONANCES:
                    T2w[tissue + component] = get_T2w(component, time_after_excitation, time_relative_inphase, field_strength)
        return T2w

    def kspace_comps_func(self, tissues, thick_kspace_comps, T2w, dephasing):
        kspace_comps = {}
        for tissue in tissues:
            if constants.TISSUES[tissue]['FF'] == .00:
                kspace_comps[tissue] = thick_kspace_comps[tissue] * T2w[tissue]
            else: # fat containing tissues
                kspace_comps[tissue + 'Water'] = thick_kspace_comps[tissue] * T2w[tissue]
                for component in constants.FAT_RESONANCES:
                    kspace_comps[tissue + component] = thick_kspace_comps[tissue] * dephasing[component] * T2w[tissue + component]
        return kspace_comps
    
    def decayed_signal_func(self, signal_level, T2w, reference_tissue, k_read_axis, k_phase_axis, freq_dir):
        return signal_level * np.take(np.take(T2w[reference_tissue], np.argmin(np.abs(k_read_axis)), axis=freq_dir), np.argmin(np.abs(k_phase_axis)))

    def noise_func(self, k_samples, noise_std):
        sampled_matrix = k_samples.shape[:-1]
        return np.random.normal(0, noise_std, sampled_matrix) + 1j * np.random.normal(0, noise_std, sampled_matrix)

    def PD_and_T1w_func(self, sequence_type, TR, TE, TI, FA, field_strength, tissues):
        return {component: get_PD_and_T1w(component, sequence_type, TR, TE, TI, FA, field_strength) for component in set(tissues).union(set(constants.FAT_RESONANCES.keys()))}

    def measured_kspace_func(self, noise, kspace_comps, FatSat, PD_and_T1w):
        measured_kspace = noise.copy()
        for component in kspace_comps:
            if 'Fat' in component:
                tissue = component[:component.find('Fat')]
                resonance = component[component.find('Fat'):]
                ratio = constants.FAT_RESONANCES[resonance]['ratio_with_FatSat' if FatSat else 'ratio']
                ratio *= constants.TISSUES[tissue]['FF']
                measured_kspace += kspace_comps[component] * PD_and_T1w[resonance] * ratio
            else:
                if 'Water' in component:
                    tissue = component[:component.find('Water')]
                    ratio = 1 - constants.TISSUES[tissue]['FF']    
                else:
                    tissue = component
                    ratio = 1.0
                measured_kspace += kspace_comps[component] * PD_and_T1w[tissue] * ratio
        return measured_kspace

    def gridded_kspace_func(self, k_grid_axes, is_radial, measured_kspace, k_samples, FOV, matrix):
        grid_shape = tuple(len(k_grid_axes[dim]) for dim in range(2))
        if not is_radial:
            return measured_kspace.reshape(grid_shape)
        samples = k_samples * FOV / matrix
        return recon.grid(measured_kspace, grid_shape, samples)
    
    def full_kspace_func(self, num_blank_lines, is_radial, gridded_kspace, phase_dir, homodyne, k_phase_axis):
        if (num_blank_lines == 0 or is_radial):
            return np.copy(gridded_kspace)
        shape_unsampled = tuple(num_blank_lines if dim==phase_dir else n for dim, n in enumerate(gridded_kspace.shape))
        full_kspace = np.append(gridded_kspace, np.zeros(shape_unsampled), axis=phase_dir) # zerofill
        if homodyne and (num_blank_lines > 0):
            full_kspace *= recon.homodyne_weights(len(k_phase_axis), num_blank_lines, phase_dir) # pre-weighting
            full_kspace += np.conjugate(np.flip(full_kspace))
        return full_kspace

    def apodized_kspace_func(self, full_kspace, do_apodize, apodization_alpha):
        apodized_kspace = full_kspace.copy()
        if do_apodize: 
            apodized_kspace *= recon.radial_Tukey(apodization_alpha, full_kspace.shape)
        return apodized_kspace

    def oversampled_recon_matrix_func(self, recon_matrix, full_k_matrix, matrix):
        oversampled_recon_matrix = recon_matrix.copy()
        for dim in range(2):
            oversampled_recon_matrix[dim] = int(np.round(recon_matrix[dim] * full_k_matrix[dim] / matrix[dim]))
        return oversampled_recon_matrix
    
    def zerofilled_kspace_func(self, apodized_kspace, oversampled_recon_matrix):
        return recon.zerofill(apodized_kspace, oversampled_recon_matrix)
    
    def image_array_func(self, oversampled_recon_matrix, full_k_matrix, recon_matrix, zerofilled_kspace):
        pixel_shifts = [0., 0.]
        sample_shifts = [0., 0.]
        for dim in range(2):
            if not oversampled_recon_matrix[dim]%2:
                pixel_shifts[dim] += 1/2 # half pixel shift for even matrixsize due to fft
            if (oversampled_recon_matrix[dim] - recon_matrix[dim])%2:
                pixel_shifts[dim] += 1/2 # half pixel shift due to cropping an odd number of pixels in image space
            if not full_k_matrix[dim]%2:
                sample_shifts[dim] += 1/2 # half sample shift for even matrixsize due to fft
                if (oversampled_recon_matrix[dim] - full_k_matrix[dim])%2:
                    sample_shifts[dim] -= 1 # sample shift for odd number of zeroes added
        image_array = recon.IFFT(zerofilled_kspace, pixel_shifts, sample_shifts)
        return recon.crop(image_array, recon_matrix)
    
    def RF_excitation_func(self, FA, is_gradient_echo):
        flip_angle = FA if is_gradient_echo else 90.
        return sequence.get_RF(flip_angle=flip_angle, time=0., dur=3., shape='hamming_sinc',  name='excitation')

    def RF_refocusing_floating_func(self, is_gradient_echo, turbo_factor):
        if is_gradient_echo:
            return None
        RF_refocusing = []
        for rf_echo in range(turbo_factor):
            RF_refocusing.append(sequence.get_RF(flip_angle=180., dur=3., shape='hamming_sinc',  name=f'refocusing{" " + str(rf_echo + 1) if turbo_factor > 1 else ""}'))
        return RF_refocusing

    def RF_inversion_floating_func(self, sequence_type):
        if not sequence_type == 'Inversion Recovery':
            return None
        return sequence.get_RF(flip_angle=180., dur=3., shape='hamming_sinc',  name='inversion')

    def RF_FatSat_floating_func(self, FatSat, field_strength):
        if not FatSat:
            return None
        return sequence.get_RF(flip_angle=90, time=0., dur=30./field_strength, shape='hamming_sinc',  name='FatSat')

    def FatSat_spoiler_floating_func(self, FatSat):
        if not FatSat:
            return None
        spoiler_area = 30. # uTs/m
        return sequence.get_gradient('slice', total_area=spoiler_area, name='FatSat spoiler', max_amp=self.max_amp, max_slew=self.max_slew)

    def slice_select_excitation_func(self, RF_excitation, slice_thickness):
        flat_dur = RF_excitation['dur_f']
        amp = RF_excitation['FWHM_f'] / (slice_thickness * constants.GYRO)
        time = 0.
        return sequence.get_gradient('slice', time, max_amp=amp, flat_dur=flat_dur, name='slice select excitation', max_slew=self.max_slew)

    def slice_select_rephaser_func(self, slice_select_excitation):
        slice_rephaser_area = -slice_select_excitation['area_f']/2
        slice_select_rephaser = sequence.get_gradient('slice', total_area=slice_rephaser_area, name='slice select rephaser', max_amp=self.max_amp, max_slew=self.max_slew)
        time = (slice_select_excitation['dur_f'] + slice_select_rephaser['dur_f']) / 2
        sequence.move_waveform(slice_select_rephaser, time)
        return slice_select_rephaser

    def slice_select_refocusing_floating_func(self, RF_refocusing_floating, slice_thickness, turbo_factor):
        if RF_refocusing_floating is None:
            return None
        flat_dur = RF_refocusing_floating[0]['dur_f']
        amp = RF_refocusing_floating[0]['FWHM_f'] / (slice_thickness * constants.GYRO)
        slice_select_refocusing = []
        for rf_echo in range(turbo_factor):
            slice_select_refocusing.append(sequence.get_gradient('slice', max_amp=amp, flat_dur=flat_dur, name='slice select refocusing', max_slew=self.max_slew))
        return slice_select_refocusing

    def slice_select_inversion_floating_func(self, sequence_type, RF_inversion_floating, slice_thickness):
        if sequence_type!='Inversion Recovery':
            return None
        flat_dur = RF_inversion_floating['dur_f']
        amp = RF_inversion_floating['FWHM_f'] / (self.inversion_thk_factor * slice_thickness * constants.GYRO)
        return sequence.get_gradient('slice', max_amp=amp, flat_dur=flat_dur, name='slice select inversion', max_slew=self.max_slew)
    
    def inversion_spoiler_floating_func(self, sequence_type):
        if sequence_type!='Inversion Recovery':
            return None
        spoiler_area = 30. # uTs/m
        return sequence.get_gradient('slice', total_area=spoiler_area, name='inversion spoiler', max_amp=self.max_amp, max_slew=self.max_slew)

    def readouts_floating_func(self, k_read_axis, pixel_bandwidth, matrix_F, FOV_F, turbo_factor, EPI_factor):
        pixel_size = (len(k_read_axis)-1) / len(k_read_axis) / (max(k_read_axis)-min(k_read_axis))
        flat_area = 1e3 / pixel_size / constants.GYRO # uTs/m
        amp = pixel_bandwidth * matrix_F / (FOV_F * constants.GYRO) # mT/m
        readouts = []
        for rf_echo in range(turbo_factor):
            readouts.append([])           
            for gr_echo in range(EPI_factor):
                suffix = ((" " if (turbo_factor > 1 or EPI_factor > 1) else "")
                        + (str(rf_echo + 1) if turbo_factor > 1 else "")
                        + ("." if (turbo_factor > 1 and EPI_factor > 1) else "")
                        + (str(gr_echo + 1) if EPI_factor > 1 else ""))
                readout = sequence.get_gradient('frequency', max_amp=amp, flat_area=flat_area, name='readout'+suffix, max_slew=self.max_slew)
                readouts[-1].append(readout)
        return readouts
    
    def sampling_windows_floating_func(self, turbo_factor, EPI_factor, readouts_floating):
        sampling_windows = []
        for rf_echo in range(turbo_factor):
            sampling_windows.append([])
            for gr_echo in range(EPI_factor):
                suffix = ((" " if (turbo_factor > 1 or EPI_factor > 1) else "")
                        + (str(rf_echo + 1) if turbo_factor > 1 else "")
                        + ("." if (turbo_factor > 1 and EPI_factor > 1) else "")
                        + (str(gr_echo + 1) if EPI_factor > 1 else ""))
                adc = sequence.get_ADC(dur=readouts_floating[0][0]['flat_dur_f'], name='sampling'+suffix)
                sampling_windows[-1].append(adc)
        return sampling_windows

    def readout_risetime_func(self, readouts_floating):
        return readouts_floating[0][0]['risetime_f']
    
    def read_prephaser_floating_func(self, readouts_floating):
        return sequence.get_gradient('frequency', total_area=readouts_floating[0][0]['area_f']/2, name='read prephaser', max_amp=self.max_amp, max_slew=self.max_slew)

    def phaser_duration_func(self, largest_phaser_area):
        largest_phaser = sequence.get_gradient('phase', total_area=largest_phaser_area, max_amp=self.max_amp, max_slew=self.max_slew)
        return largest_phaser['dur_f']
        
    def max_blip_dur_func(self, EPI_factor, phase_step_area, num_shots, turbo_factor):
        if (EPI_factor <= 1):
            return 0
        max_blip_area = phase_step_area * num_shots * turbo_factor
        max_blip = sequence.get_gradient('phase', total_area=max_blip_area, max_amp=self.max_amp, max_slew=self.max_slew)
        return max_blip['dur_f']
    
    def phasers_floating_func(self, turbo_factor, largest_phaser_area, pe_table, phase_step_area, shot):
        phasers = []
        for rf_echo in range(turbo_factor):
            phaser_area = largest_phaser_area + pe_table[shot-1, rf_echo, 0] * phase_step_area
            suffix = f' {rf_echo + 1}' if turbo_factor > 1 else ''
            phaser = sequence.get_gradient('phase', total_area=largest_phaser_area, name='phase encode'+suffix, max_amp=self.max_amp, max_slew=self.max_slew)
            if abs(largest_phaser_area) > 1e-5:
                sequence.rescale_gradient(phaser, phaser_area / largest_phaser_area)
            phasers.append(phaser)
        return phasers
    
    def blips_floating_func(self, turbo_factor, EPI_factor, phase_step_area, pe_table, shot):
        blips = []
        for rf_echo in range(turbo_factor):
            blips.append([])
            for gr_echo in range(1, EPI_factor):
                blip_area = phase_step_area * (pe_table[shot-1, rf_echo, gr_echo] - pe_table[shot-1, rf_echo, gr_echo-1])
                blip = sequence.get_gradient('phase', total_area=blip_area, name='blip', max_amp=self.max_amp, max_slew=self.max_slew)
                blips[-1].append(blip)
        return blips

    def rephasers_floating_func(self, turbo_factor, phasers_floating, blips_floating, largest_phaser_area):
        rephasers = []
        for rf_echo in range(turbo_factor):
            suffix = f' {rf_echo + 1}' if turbo_factor > 1 else ''
            rephaser_area = -phasers_floating[rf_echo]['area_f']
            for blip in blips_floating[rf_echo]:
                rephaser_area -= blip['area_f']
            rephaser = sequence.get_gradient('phase', total_area=largest_phaser_area, name='rephaser'+suffix, max_amp=self.max_amp, max_slew=self.max_slew)
            if abs(largest_phaser_area) > 1e-5:
                sequence.rescale_gradient(rephaser, rephaser_area / largest_phaser_area)
            rephasers.append(rephaser)
        return rephasers

    def spoiler_floating_func(self):
        spoiler_area = 30. # uTs/m
        return sequence.get_gradient('slice', total_area=spoiler_area, name='spoiler', max_amp=self.max_amp, max_slew=self.max_slew)

    def slice_select_refocusing_func(self, slice_select_refocusing_floating, readtrain_spacing):
        if slice_select_refocusing_floating is None:
            return None
        slice_select_refocusing = []
        for rf_echo, grad in enumerate(slice_select_refocusing_floating):
            slice_select_refocusing.append(grad)
            time = get_readtrain_pos(readtrain_spacing, rf_echo) - readtrain_spacing/2
            sequence.move_waveform(slice_select_refocusing[rf_echo], time)
        return slice_select_refocusing

    def RF_refocusing_func(self, RF_refocusing_floating, readtrain_spacing):
        if RF_refocusing_floating is None:
            return None
        RF_refocusing = []
        for rf_echo, RF in enumerate(RF_refocusing_floating):
            RF_refocusing.append(RF)
            time = get_readtrain_pos(readtrain_spacing, rf_echo) - readtrain_spacing/2
            sequence.move_waveform(RF_refocusing[rf_echo], time)
        return RF_refocusing

    def slice_select_inversion_func(self, slice_select_inversion_floating, TI):
        if slice_select_inversion_floating is None:
            return None
        slice_select_inversion = slice_select_inversion_floating
        sequence.move_waveform(slice_select_inversion, -TI)
        return slice_select_inversion
        
    def RF_inversion_func(self, RF_inversion_floating, TI):
        if RF_inversion_floating is None:
            return None
        RF_inversion = RF_inversion_floating
        sequence.move_waveform(RF_inversion, -TI)
        return RF_inversion
    
    def inversion_spoiler_func(self, inversion_spoiler_floating, RF_inversion):
        if inversion_spoiler_floating is None:
            return None
        inversion_spoiler = inversion_spoiler_floating
        time = RF_inversion['time'][-1] + inversion_spoiler['dur_f']/2
        sequence.move_waveform(inversion_spoiler, time)
        return inversion_spoiler

    def FatSat_spoiler_func(self, FatSat_spoiler_floating, slice_select_excitation):
        if FatSat_spoiler_floating is None:
            return None
        FatSat_spoiler = FatSat_spoiler_floating
        time = slice_select_excitation['time'][0] - FatSat_spoiler['dur_f']/2
        sequence.move_waveform(FatSat_spoiler, time)
        return FatSat_spoiler

    def RF_FatSat_func(self, RF_FatSat_floating, FatSat_spoiler_floating):
        if RF_FatSat_floating is None:
            return None
        RF_FatSat = RF_FatSat_floating
        t = FatSat_spoiler_floating['time'][0] - RF_FatSat['dur_f']/2
        sequence.move_waveform(RF_FatSat, t)
        return RF_FatSat

    def readouts_func(self, turbo_factor, readtrain_spacing, EPI_factor, gr_echo_spacing, readouts_floating):
        readouts = []
        for rf_echo in range(turbo_factor):
            readouts.append([])
            readtrain_pos = get_readtrain_pos(readtrain_spacing, rf_echo)
            for gr_echo in range(EPI_factor):
                readout = readouts_floating[rf_echo][gr_echo]
                pos = readtrain_pos + (gr_echo - (EPI_factor-1) / 2) * gr_echo_spacing
                sequence.move_waveform(readout, pos)
                if gr_echo%2 and readout['area_f'] > 0:
                    sequence.rescale_gradient(readout, -1)
                readouts[-1].append(readout)
        return readouts
                    
    def sampling_windows_func(self, turbo_factor, readtrain_spacing, EPI_factor, gr_echo_spacing, sampling_windows_floating):
        sampling_windows = []
        for rf_echo in range(turbo_factor):
            readtrain_pos = get_readtrain_pos(readtrain_spacing, rf_echo)
            sampling_windows.append([])
            for gr_echo in range(EPI_factor):
                sampling_window = sampling_windows_floating[rf_echo][gr_echo]
                pos = readtrain_pos + (gr_echo - (EPI_factor-1) / 2) * gr_echo_spacing
                sequence.move_waveform(sampling_window, pos)
                sampling_windows[-1].append(sampling_window)
        return sampling_windows
    
    def read_prephaser_func(self, read_prephaser_floating, is_gradient_echo, readouts, RF_excitation):
        read_prephaser = read_prephaser_floating
                
        if is_gradient_echo:
            if read_prephaser['area_f'] > 0:
                sequence.rescale_gradient(read_prephaser, -1)
            first_readout = readouts[0][0]
            prephase_time = first_readout['center_f'] - sum([grad['dur_f'] for grad in [read_prephaser, first_readout]])/2
        else:
            if read_prephaser['area_f'] < 0:
                sequence.rescale_gradient(read_prephaser, -1)
            prephase_time = sum([object['dur_f'] for object in [RF_excitation, read_prephaser]])/2
        sequence.move_waveform(read_prephaser, prephase_time)
        return read_prephaser
    
    def phasers_func(self, turbo_factor, readtrain_spacing, phasers_floating, gre_echo_train_dur, readout_risetime):
        phasers = []
        for rf_echo in range(turbo_factor):
            phaser = phasers_floating[rf_echo]
            readtrain_pos = get_readtrain_pos(readtrain_spacing, rf_echo)
            phaser_time = readtrain_pos - (gre_echo_train_dur + phaser['dur_f'])/2 + readout_risetime
            sequence.move_waveform(phaser, phaser_time)
            phasers.append(phaser)
        return phasers
    
    def rephasers_func(self, turbo_factor, readtrain_spacing, gre_echo_train_dur, readout_risetime, rephasers_floating):
        rephasers = []
        for rf_echo in range(turbo_factor):
            readtrain_pos = get_readtrain_pos(readtrain_spacing, rf_echo)
            rephaser = rephasers_floating[rf_echo]
            rephaser_time = readtrain_pos + (gre_echo_train_dur + rephaser['dur_f'])/2 - readout_risetime
            sequence.move_waveform(rephaser, rephaser_time)
            rephasers.append(rephaser)
        return rephasers
        
    def blips_func(self, turbo_factor, readtrain_spacing, EPI_factor, gr_echo_spacing, blips_floating):
        blips = []
        for rf_echo in range(turbo_factor):
            readtrain_pos = get_readtrain_pos(readtrain_spacing, rf_echo)
            blips.append([])
            for gr_echo in range(EPI_factor-1):
                blip = blips_floating[rf_echo][gr_echo]
                blip_time = readtrain_pos + gr_echo_spacing * (gr_echo - EPI_factor/2 + 1)
                sequence.move_waveform(blip, blip_time)
                blips[-1].append(blip)
        return blips

    def spoiler_func(self, readouts, spoiler_floating):
        spoiler = spoiler_floating
        spoiler_time = readouts[-1][-1]['center_f'] + (readouts[-1][-1]['flat_dur_f'] + spoiler['dur_f']) / 2
        sequence.move_waveform(spoiler, spoiler_time)
        return spoiler

    def sequence_start_func(self, sequence_type, slice_select_inversion, RF_FatSat, slice_select_excitation):
        if sequence_type == 'Inversion Recovery': 
            return slice_select_inversion['time'][0]
        elif self.FatSat:
            return RF_FatSat['time'][0]
        else:
            return slice_select_excitation['time'][0]

    def signal_curves_func(self, measured_kspace, shot, is_radial, turbo_factor, EPI_factor, pe_table, phase_dir, time_after_excitation):
        signal_curves = []
        scale = 1 / np.max(np.abs(np.real(measured_kspace)))
        signal_exponent = .5
        spoke = shot-1 if is_radial else 0
        for rf_echo in range(turbo_factor):
            signal_curves.append([])
            for gr_echo in range(EPI_factor):
                ky = pe_table[shot-1, rf_echo, gr_echo]
                waveform = np.real(np.take(measured_kspace[..., spoke], indices=ky, axis=phase_dir))
                t = np.take(time_after_excitation[..., spoke if spoke<time_after_excitation.shape[-1] else 0], indices=ky, axis=phase_dir)
                signal = sequence.get_signal(waveform, t, scale, signal_exponent)
                signal_curves[-1].append(signal)
        return signal_curves

    def get_hover_tool(self, board, attributes):
        with open(Path(__file__).parent / 'hoverCallback.js', 'r') as file:
            hover_callback = CustomJS(args={'hover_index': self.hover_index, 'board': board}, code=file.read())
        if board == 'slice':
            hover_callback = None
        return HoverTool(tooltips=[(attr, f'@{attr}') for attr in attributes], attachment='below', callback=hover_callback)
    
    def update_k_line_coords(self, attr, old, hover_index):
        if len(hover_index['index']) == 0:
            self.k_line.event(coords=[None])
            return
        board = hover_index['board'][0]
        index = hover_index['index'][0]
        object = self.graph[f'{board}_objects'].value[index]
        k_trajectory = self.graph['k_trajectory'].value
        self.k_line.event(coords=list(get_k_on_interval(object['time'][[0, -1]], k_trajectory)))

    def k_trajectory_func(self, RF_refocusing, frequency_board, phase_board, is_radial, phase_dir, spoke_angle):
        frequency_area = frequency_board[1] # TODO: avoid hard-coding
        phase_area = phase_board[1] # TODO: avoid hard-coding
        dt = .01
        refocus_intervals = [list(rf['time'][[0, -1]]) for rf in RF_refocusing] if RF_refocusing else []
        t = np.concatenate((*(area['time'] for area in [frequency_area, phase_area]), [t for ref in refocus_intervals for t in ref])) # k event times
        t = np.unique(np.concatenate((t, np.arange(0., max(t), dt)))) # merge with time grid
        kx = get_k_coords(t, *(frequency_area[dim] for dim in ['G read', 'time']), refocus_intervals)
        ky = get_k_coords(t, *(phase_area[dim] for dim in ['G phase', 'time']), refocus_intervals)
        if not is_radial:
            if phase_dir==1:
                kx, ky = ky, kx
        else: # rotate by spoke/blade angle
            angle = np.radians(spoke_angle)
            cos, sin = np.cos(angle), np.sin(angle)
            kx, ky = cos * kx - sin * ky, sin * kx + cos * ky
        return {'kx': kx, 'ky': ky, 't': t, 'dt': dt}

    def time_dim_func(self):
        return hv.Dimension('time', label='time', unit='ms')
    
    def frequency_dim_func(self):
        return hv.Dimension('frequency', label='G read', unit='mT/m', range=(-30, 30))
    
    def phase_dim_func(self):
        return hv.Dimension('phase', label='G phase', unit='mT/m', range=(-30, 30))
    
    def slice_dim_func(self):
        return hv.Dimension('slice', label='G slice', unit='mT/m', range=(-30, 30))
        
    def RF_dim_func(self):
        return hv.Dimension('RF', label='RF', unit='μT', range=(-5, 25))
    
    def signal_dim_func(self):
        return hv.Dimension('signal', label='signal', unit='a.u.', range=(-1, 1))
    
    def ADC_dim_func(self):
        return hv.Dimension('ADC', label='ADC', unit='')
    
    def frequency_objects_func(self, read_prephaser, readouts):
        objects = [read_prephaser, *flatten_dicts(readouts)]
        return [obj for obj in objects if obj]
    
    def phase_objects_func(self, phasers, rephasers, blips):
        objects = [*flatten_dicts(phasers), *flatten_dicts(rephasers), *flatten_dicts(blips)]
        return [obj for obj in objects if obj]
        
    def slice_objects_func(self, slice_select_inversion, inversion_spoiler, FatSat_spoiler, slice_select_excitation, slice_select_rephaser, slice_select_refocusing, spoiler):
        objects = [slice_select_inversion, inversion_spoiler, FatSat_spoiler, slice_select_excitation, slice_select_rephaser, *flatten_dicts(slice_select_refocusing), spoiler]
        return [obj for obj in objects if obj]
    
    def RF_objects_func(self, RF_inversion, RF_FatSat, RF_excitation, RF_refocusing):
        objects = [RF_inversion, RF_FatSat, RF_excitation, *flatten_dicts(RF_refocusing)]
        return [obj for obj in objects if obj]
    
    def signal_objects_func(self, signal_curves):
        return flatten_dicts(signal_curves)
    
    def ADC_objects_func(self, sampling_windows):
        objects = flatten_dicts(sampling_windows)
        for obj in objects:
            obj.update({'c1': obj['time'][0], 'c2': -2, 'c3': obj['time'][-1], 'c4': 2})
        return objects
    
    def TR_span_func(self, sequence_start, TR, time_dim, frequency_dim, phase_dim, slice_dim, RF_dim, signal_dim):
        TR_span = {}
        for board_dim in [frequency_dim, phase_dim, slice_dim, RF_dim, signal_dim]:
            TR_span[board_dim.name] = hv.VSpan(-20000, sequence_start, kdims=[time_dim, board_dim]).opts(color='gray', fill_alpha=.3)
            TR_span[board_dim.name] *= hv.VSpan(sequence_start + TR, 20000, kdims=[time_dim, board_dim]).opts(color='gray', fill_alpha=.3)
        return TR_span

    def frequency_hover_func(self):
        return self.get_hover_tool('frequency', ['name', 'center', 'duration', 'area'])
    
    def phase_hover_func(self):
        return self.get_hover_tool('phase', ['name', 'center', 'duration', 'area'])
    
    def slice_hover_func(self):
        return self.get_hover_tool('slice', ['name', 'center', 'duration', 'area'])
    
    def RF_hover_func(self):
        return self.get_hover_tool('RF', ['name', 'center', 'duration', 'flip_angle'])
    
    def signal_hover_func(self):
        return self.get_hover_tool('signal', ['name', 'center', 'duration'])
    
    def frequency_board_func(self, time_dim, frequency_dim, frequency_objects, TR_span, frequency_hover):
        vdims = [tip[0] for tip in frequency_hover.tooltips]
        specs = [hline(time_dim, frequency_dim),
                 hv.Area(sequence.accumulate_waveforms(frequency_objects, 'frequency'), time_dim, frequency_dim).opts(color=BOARD_COLORS['frequency']),
                 hv.Polygons(frequency_objects, kdims=[time_dim, frequency_dim], vdims=vdims).opts(tools=[frequency_hover], cmap=[BOARD_COLORS['frequency']], hooks=[hideframe_hook, partial(bounds_hook, xbounds=(-19000, 19000))]),
                 TR_span['frequency']]
        return specs
        
    def phase_board_func(self, time_dim, phase_dim, phase_objects, TR_span, phase_hover):
        vdims = [tip[0] for tip in phase_hover.tooltips]
        specs = [hline(time_dim, phase_dim),
                 hv.Area(sequence.accumulate_waveforms(phase_objects, 'phase'), time_dim, phase_dim).opts(color=BOARD_COLORS['phase']),
                 hv.Polygons(phase_objects, kdims=[time_dim, phase_dim], vdims=vdims).opts(tools=[phase_hover], cmap=[BOARD_COLORS['phase']], hooks=[hideframe_hook, partial(bounds_hook, xbounds=(-19000, 19000))]),
                 TR_span['phase']]
        return specs
    
    def slice_board_func(self, time_dim, slice_dim, slice_objects, TR_span, slice_hover):
        vdims = [tip[0] for tip in slice_hover.tooltips]
        specs = [hline(time_dim, slice_dim),
                 hv.Area(sequence.accumulate_waveforms(slice_objects, 'slice'), time_dim, slice_dim).opts(color=BOARD_COLORS['slice']),
                 hv.Polygons(slice_objects, kdims=[time_dim, slice_dim], vdims=vdims).opts(tools=[slice_hover], cmap=[BOARD_COLORS['slice']], hooks=[hideframe_hook, partial(bounds_hook, xbounds=(-19000, 19000))]),
                 TR_span['slice']]
        return specs
    
    def RF_board_func(self, time_dim, RF_dim, RF_objects, TR_span, RF_hover):
        vdims = [tip[0] for tip in RF_hover.tooltips]
        specs = [hline(time_dim, RF_dim),
                 hv.Area(sequence.accumulate_waveforms(RF_objects, 'RF'), time_dim, RF_dim).opts(color=BOARD_COLORS['RF']),
                 hv.Polygons(RF_objects, kdims=[time_dim, RF_dim], vdims=vdims).opts(tools=[RF_hover], cmap=[BOARD_COLORS['RF']], hooks=[hideframe_hook, partial(bounds_hook, xbounds=(-19000, 19000))]),
                 TR_span['RF']]
        return specs
    
    def signal_board_func(self, time_dim, signal_dim, signal_objects, ADC_objects, TR_span, signal_hover):
        vdims = [tip[0] for tip in signal_hover.tooltips]
        specs = [hline(time_dim, signal_dim),
                 hv.Area(sequence.accumulate_waveforms(signal_objects, 'signal'), time_dim, signal_dim).opts(color=BOARD_COLORS['signal']),
                 hv.Polygons(signal_objects, kdims=[time_dim, signal_dim], vdims='signal').opts(tools=[], cmap=[BOARD_COLORS['signal']], hooks=[hideframe_hook, partial(bounds_hook, xbounds=(-19000, 19000))]),
                 hv.Rectangles(ADC_objects, kdims=['c1', 'c2', 'c3', 'c4'], vdims=vdims).opts(tools=[signal_hover]),
                 TR_span['signal']]
        return specs

    def sequence_plot_func(self, frequency_board, phase_board, slice_board, RF_board, signal_board):
        boards = [frequency_board, phase_board, slice_board, RF_board, signal_board]
        board_plots = []
        for board in boards[:-1]:
            board_plots.append(hv.Overlay(board).opts(width=1700, height=120, border=0, xaxis=None))
        board_plots.append(hv.Overlay(boards[-1]).opts(width=1700, height=180, border=0, xaxis='bottom'))
        return hv.Layout(list(board_plots)).cols(1).options(toolbar='below')

    def kspace_func(self, kspace_type, show_processed_kspace, oversampled_recon_matrix, FOV, recon_matrix, full_k_matrix, zerofilled_kspace, kspace_exponent, gridded_kspace, k_grid_axes):
        operator = constants.OPERATORS[kspace_type]
        if show_processed_kspace:
            k_axes = []
            for dim in range(2):
                k_axes.append(recon.get_k_axis(oversampled_recon_matrix[dim], FOV[dim] / recon_matrix[dim]))
                # half-sample shift axis when odd number of zeroes:
                if (oversampled_recon_matrix[dim] - full_k_matrix[dim])%2:
                    shift = recon_matrix[dim] / (2 * oversampled_recon_matrix[dim] * FOV[dim])
                    k_axes[-1] -= shift
            ksp = xr.DataArray(
                operator(zerofilled_kspace**kspace_exponent), 
                dims=('ky', 'kx'),
                coords={'kx': k_axes[1], 'ky': k_axes[0]}
            )
        else:
            ksp = xr.DataArray(
                operator(gridded_kspace**kspace_exponent), 
                dims=('ky', 'kx'),
                coords={'kx': k_grid_axes[1], 'ky': k_grid_axes[0]}
            )
        ksp.kx.attrs['units'] = ksp.ky.attrs['units'] = '1/mm'
        lim = 1.12 * max(k_grid_axes[1])
        return hv.Image(ksp, vdims=['magnitude']).opts(xlim=(-lim,lim), ylim=(-lim,lim))

    def FOV_box_func(self, show_FOV, is_radial, FOV, matrix, freq_dir, phase_dir, k_read_axis, k_phase_axis):
        if not show_FOV:
            return hv.Overlay([])
        rec_FOV_shape = hv.Box(0, 0, tuple(FOV[::-1])).opts(color='yellow')
        if is_radial:
            radial_FOV = FOV[freq_dir] * len(k_read_axis) / matrix[freq_dir]
            acq_FOV_shape = hv.Ellipse(0, 0, radial_FOV).opts(line_color='lightblue')
        else:
            acq_FOV = FOV.copy()
            acq_FOV[phase_dir] *= len(k_phase_axis) / matrix[phase_dir]
            acq_FOV_shape = hv.Box(0, 0, tuple(acq_FOV[::-1])).opts(color='lightblue')
        return acq_FOV_shape * rec_FOV_shape
    
    def image_func(self, image_type, recon_matrix, FOV, image_array):
        operator = constants.OPERATORS[image_type]
        axes = [(np.arange(recon_matrix[dim]) - (recon_matrix[dim]-1)/2) / recon_matrix[dim] * FOV[dim] for dim in range(2)]
        img = xr.DataArray(
            operator(image_array), 
            dims=('y', 'x'),
            coords={'x': axes[1], 'y': axes[0][::-1]}
        )
        img.x.attrs['units'] = img.y.attrs['units'] = 'mm'
        return hv.Overlay([hv.Image(img, vdims=['magnitude'])])
    
    def set_reference_SNR(self, event=None):
        self.reference_SNR = self.graph['SNR'].value
    
    @param.depends('sequence_type', 'FatSat', 'TR', 'TE', 'FA', 'TI', 'FOV_F', 'FOV_P', 'phase_oversampling', 'num_shots', 'matrix_F', 'matrix_P', 'slice_thickness', 'trajectory', 'frequency_direction', 'pixel_bandwidth', 'partial_Fourier', 'turbo_factor', 'EPI_factor', 'shot')
    def display_sequence_plot(self):
        return self.graph['sequence_plot'].value
    
    @param.depends('object', 'field_strength', 'sequence_type', 'FatSat', 'TR', 'TE', 'FA', 'TI', 'FOV_F', 'FOV_P', 'phase_oversampling', 'num_shots', 'matrix_F', 'matrix_P', 'recon_matrix_F', 'recon_matrix_P', 'slice_thickness', 'trajectory', 'frequency_direction', 'pixel_bandwidth', 'NSA', 'partial_Fourier', 'turbo_factor', 'EPI_factor', 'kspace_type', 'show_processed_kspace', 'kspace_exponent', 'homodyne', 'do_apodize', 'apodization_alpha', 'do_zerofill', 'radial_FOV_oversampling')
    def display_kspace(self):
        return self.graph['kspace'].value

    @param.depends('object', 'field_strength', 'sequence_type', 'FatSat', 'TR', 'TE', 'FA', 'TI', 'FOV_F', 'FOV_P', 'phase_oversampling', 'num_shots', 'matrix_F', 'matrix_P', 'recon_matrix_F', 'recon_matrix_P', 'slice_thickness', 'trajectory', 'frequency_direction', 'pixel_bandwidth', 'NSA', 'partial_Fourier', 'turbo_factor', 'EPI_factor', 'image_type', 'show_FOV', 'homodyne', 'do_apodize', 'apodization_alpha', 'do_zerofill', 'radial_FOV_oversampling')
    def display_image(self):
        return self.graph['image'].value * self.graph['FOV_box'].value