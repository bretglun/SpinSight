import holoviews as hv
from holoviews import streams
import param
import numpy as np
import math
from pathlib import Path
import xarray as xr
from spinsight import constants, sequence, recon, phantom
from spinsight.DAG import InputParamNode, ComputeNode, ActionNode, OutputParamNode
from bokeh.models import HoverTool, CustomJS, ColumnDataSource
from functools import partial
import warnings

hv.extension('bokeh')


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


def add_to_pipeline(pipeline, functions):
    pipeline.update({f: True for f in functions})


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
    num_shots_label = param.String('# shots')
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
    
    sequence = param.ObjectSelector(default=constants.SEQUENCES[0], precedence=1, label='Pulse sequence')
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
        self._initialized = False
        
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

        self.time_dim = hv.Dimension('time', label='time', unit='ms')

        self.boards = { 'frequency': {'dim': hv.Dimension('frequency', label='G read', unit='mT/m', range=(-30, 30)), 'color': 'cadetblue'}, 
                        'phase': {'dim': hv.Dimension('phase', label='G phase', unit='mT/m', range=(-30, 30)), 'color': 'cadetblue'}, 
                        'slice': {'dim': hv.Dimension('slice', label='G slice', unit='mT/m', range=(-30, 30)), 'color': 'cadetblue'}, 
                        'RF': {'dim': hv.Dimension('RF', label='RF', unit='μT', range=(-5, 25)), 'color': 'red'},
                        'signal': {'dim': hv.Dimension('signal', label='signal', unit='a.u.', range=(-1, 1)), 'color': 'orange'},
                        'ADC': {'dim': hv.Dimension('ADC', label='ADC', unit=''), 'color': 'peru'} }
        
        self.board_plots = {board: {'hline': hv.HLine(0.0, kdims=[self.time_dim, self.boards[board]['dim']]).opts(tools=['xwheel_zoom', 'xpan', 'reset'], default_tools=[], active_tools=['xwheel_zoom', 'xpan'])} for board in self.boards if board != 'ADC'}

        hv.opts.defaults(hv.opts.Image(width=500, height=500, invert_yaxis=False, toolbar='below', cmap='gray', aspect='equal'))
        hv.opts.defaults(hv.opts.HLine(line_width=1.5, line_color='gray'))
        hv.opts.defaults(hv.opts.VSpan(color='orange', fill_alpha=.1, hover_fill_alpha=.8, default_tools=[]))
        hv.opts.defaults(hv.opts.Rectangles(color=self.boards['ADC']['color'], line_color=self.boards['ADC']['color'], fill_alpha=.1, line_alpha=.3, hover_fill_alpha=.8, default_tools=[]))
        hv.opts.defaults(hv.opts.Box(line_width=3))
        hv.opts.defaults(hv.opts.Ellipse(line_width=3))
        hv.opts.defaults(hv.opts.Area(fill_alpha=.5, line_width=1.5, line_color='gray', default_tools=[]))
        hv.opts.defaults(hv.opts.Polygons(line_width=1.5, fill_alpha=0, line_alpha=0, line_color='gray', selection_line_color='black', hover_fill_alpha=.8, hover_line_alpha=1, selection_fill_alpha=.8, selection_line_alpha=1, nonselection_line_alpha=0, default_tools=[]))
        hv.opts.defaults(hv.opts.Curve(line_width=5, line_color='peru'))
        hv.opts.defaults(hv.opts.Points(line_color=None, color='peru', size=15))

        self.max_amp = 25. # mT/m
        self.max_slew = 80. # T/m/s
        self.inversion_thk_factor = 1.1 # make inversion slice 10% thicker

        for board in self.boards:
            self.boards[board]['objects'] = {}

        # create InputParamNodes for all params
        for par in self.param:
            if par == 'name':
                continue
            setattr(self, f'{par}_node', InputParamNode(self, par))
            #self.param.watch(lambda _: getattr(self, f'_watch_{par}')(), par, precedence=1)

        self.phantom = ComputeNode(
            lambda object:
            phantom.load(object, self.min_voxel_size),
            [self.object_node]
        )
        
        self.tissues = ComputeNode(
            lambda phantom:
            list(phantom['shapes'].keys()),
            [self.phantom]
        )

        self.is_radial = ComputeNode(
            lambda trajectory: 
            trajectory in ['Radial', 'PROPELLER'], 
            [self.trajectory_node]
        )
        
        self.is_gradient_echo = ComputeNode(
            lambda sequence: 
            'Gradient Echo' in sequence, 
            [self.sequence_node]
        )

        self.freq_dir = ComputeNode(
            lambda frequency_direction, is_radial: constants.DIRECTIONS[frequency_direction] if not is_radial else 1,
            [self.frequency_direction_node, self.is_radial]
        )

        self.phase_dir = ComputeNode(
            lambda freq_dir: 1 - freq_dir,
            [self.freq_dir]
        )

        self.FOV = ComputeNode(
            lambda FOV_F, FOV_P, freq_dir: [FOV_P, FOV_F] if freq_dir else [FOV_F, FOV_P],
            [self.FOV_F_node, self.FOV_P_node, self.freq_dir]
        )
        
        self.matrix = ComputeNode(
            lambda matrix_F, matrix_P, freq_dir: [matrix_P, matrix_F] if freq_dir else [matrix_F, matrix_P],
            [self.matrix_F_node, self.matrix_P_node, self.freq_dir]
        )

        self.recon_matrix = ComputeNode(
            lambda recon_matrix_P, recon_matrix_F, freq_dir, do_zerofill, matrix:
            ([recon_matrix_P, recon_matrix_F] if freq_dir else [recon_matrix_F, recon_matrix_P]) if do_zerofill else matrix,
            [self.recon_matrix_P_node, self.recon_matrix_F_node, self.freq_dir, self.do_zerofill_node, self.matrix]
        )

        self.RF_excitation = ComputeNode(
            self.RF_excitation_func,
            [self.FA_node, self.is_gradient_echo]
        )

        self.RF_refocusing_floating = ComputeNode(
            self.RF_refocusing_floating_func,
            [self.is_gradient_echo, self.turbo_factor_node]
        )

        self.RF_inversion_floating = ComputeNode(
            self.RF_inversion_floating_func,
            [self.sequence_node]
        )

        self.RF_FatSat_floating = ComputeNode(
            self.RF_FatSat_floating_func,
            [self.FatSat_node, self.field_strength_node]
        )

        self.FatSat_spoiler_floating = ComputeNode(
            self.FatSat_spoiler_floating_func,
            [self.FatSat_node]
        )

        self.slice_select_excitation = ComputeNode(
            self.slice_select_excitation_func,
            [self.RF_excitation, self.slice_thickness_node]
        )

        self.slice_select_rephaser = ComputeNode(
            self.slice_select_rephaser_func,
            [self.slice_select_excitation]
        )

        self.slice_select_refocusing_floating = ComputeNode(
            self.slice_select_refocusing_floating_func,
            [self.RF_refocusing_floating, self.slice_thickness_node, self.turbo_factor_node]
        )

        self.slice_select_inversion_floating = ComputeNode(
            self.slice_select_inversion_floating_func,
            [self.sequence_node, self.RF_inversion_floating, self.slice_thickness_node]
        )

        self.inversion_spoiler_floating = ComputeNode(
            self.inversion_spoiler_floating_func,
            [self.sequence_node]
        )

        self.readouts_floating = ComputeNode(
            self.readouts_floating_func,
            [self.k_read_axis, self.pixel_bandwidth_node, self.matrix_F_node, self.FOV_F_node, self.turbo_factor_node, self.EPI_factor_node]
        )
        
        self.sampling_windows_floating = ComputeNode(
            self.sampling_windows_floating_func,
            [self.turbo_factor_node, self.EPI_factor_node, self.readouts_floating]
        )

        self.readout_risetime = ComputeNode(
            self.readout_risetime_func,
            [self.readouts_floating]
        )
        
        self.read_prephaser_floating = ComputeNode(
            self.read_prephaser_floating_func,
            [self.readouts_floating]
        )
        
        self.phase_step_area = ComputeNode(
            # uTs/m
            lambda k_phase_axis:
            np.mean(np.diff(k_phase_axis)) * 1e3 / constants.GYRO,
            [self.k_phase_axis]
        )

        self.largest_phaser_area = ComputeNode(
            # uTs/m
            lambda k_phase_axis:
            np.min(k_phase_axis) * 1e3 / constants.GYRO,
            [self.k_phase_axis]
        )

        self.phaser_duration = ComputeNode(
            self.phaser_duration_func,
            [self.largest_phaser_area]
        )
        
        self.max_blip_dur = ComputeNode(
            self.max_blip_dur_func,
            [self.EPI_factor_node, self.phase_step_area, self.num_shots, self.turbo_factor_node]
        )
        
        self.readout_gap = ComputeNode(
            lambda max_blip_dur, readouts:
            max(max_blip_dur - 2 * readouts[0][0]['risetime_f'], 0),
            [self.max_blip_dur, self.readouts_floating]
        )
        
        self.gr_echo_spacing = ComputeNode(
            lambda readouts, readout_gap:
            readouts[0][0]['dur_f'] + readout_gap,
            [self.readouts_floating, self.readout_gap]
        )

        self.gre_echo_train_dur = ComputeNode(
            lambda EPI_factor, gr_echo_spacing, readout_gap:
            EPI_factor * gr_echo_spacing - readout_gap,
            [self.EPI_factor_node, self.gr_echo_spacing, self.readout_gap]
        )

        self.phasers_floating = ComputeNode(
            self.phasers_floating_func,
            [self.turbo_factor_node, self.largest_phaser_area, self.pe_table, self.phase_step_area, self.shot_node]
        )
        
        self.blips_floating = ComputeNode(
            self.blips_floating_func,
            [self.turbo_factor_node, self.EPI_factor_node, self.phase_step_area, self.pe_table, self.shot_node]
        )
        
        self.rephasers_floating = ComputeNode(
            self.rephasers_floating_func,
            [self.turbo_factor_node, self.phasers_floating, self.blips_floating, self.largest_phaser_area]
        )
        
        self.spoiler_floating = ComputeNode(
            self.spoiler_floating_func,
            []
        )

        self.slice_select_refocusing = ComputeNode(
            self.slice_select_refocusing_func,
            [self.slice_select_refocusing_floating, self.readtrain_spacing]
        )

        self.RF_refocusing = ComputeNode(
            self.RF_refocusing_func,
            [self.RF_refocusing_floating, self.readtrain_spacing]
        )
        
        self.slice_select_inversion = ComputeNode(
            self.slice_select_inversion_func,
            [self.slice_select_inversion_floating, self.TI_node]
        )
        
        self.RF_inversion = ComputeNode(
            self.RF_inversion_func,
            [self.RF_inversion_floating, self.TI_node]
        )

        self.inversion_spoiler = ComputeNode(
            self.inversion_spoiler_func,
            [self.inversion_spoiler_floating, self.RF_inversion]
        )
        
        self.FatSat_spoiler = ComputeNode(
            self.FatSat_spoiler_func,
            [self.FatSat_spoiler_floating, self.slice_select_excitation]
        )
        
        self.RF_FatSat = ComputeNode(
            self.RF_FatSat_func,
            [self.RF_FatSat_floating, self.FatSat_spoiler_floating]
        )

        self.readouts = ComputeNode(
            self.readouts_func,
            [self.turbo_factor_node, self.readtrain_spacing, self.EPI_factor_node, self.gr_echo_spacing, self.readouts_floating]
        )

        self.readouts = ComputeNode(
            self.readouts_func,
            [self.turbo_factor_node, self.readtrain_spacing, self.EPI_factor_node, self.gr_echo_spacing, self.readouts_floating]
        )
        
        self.sampling_windows = ComputeNode(
            self.sampling_windows_func,
            [self.turbo_factor_node, self.readtrain_spacing, self.EPI_factor_node, self.gr_echo_spacing, self.sampling_windows_floating]
        )
        
        self.read_prephaser = ComputeNode(
            self.read_prephaser_func,
            [self.read_prephaser_floating, self.is_gradient_echo, self.readouts, self.RF_excitation]
        )
        
        self.phasers = ComputeNode(
            self.phasers_func,
            [self.turbo_factor_node, self.readtrain_spacing, self.phasers_floating, self.gre_echo_train_dur, self.readout_risetime]
        )
        
        self.rephasers = ComputeNode(
            self.rephasers_func,
            [self.turbo_factor_node, self.readtrain_spacing, self.gre_echo_train_dur, self.readout_risetime, self.rephasers_floating]
        )
        
        self.blips = ComputeNode(
            self.blips_func,
            [self.turbo_factor_node, self.readtrain_spacing, self.EPI_factor_node, self.gr_echo_spacing, self.blips_floating]
        )

        self.spoiler = ComputeNode(
            self.spoiler_func,
            [self.readouts, self.spoiler_floating]
        )

        self.centermost_echoes_linear_order = ComputeNode(
            self.centermost_echoes_linear_order_func,
            [self.central_segments, self.reverse_linear_order, self.num_segm, self.turbo_factor_node]
        )

        self.readtrain_spacing_linear_order = ComputeNode(
            self.readtrain_spacing_linear_order_func,
            [self.centermost_echoes_linear_order, self.gr_echo_spacing, self.EPI_factor_node, self.TE_node]
        )

        self.k_read_axis = ComputeNode(
            self.k_read_axis_func,
            [self.freq_dir, self.FOV, self.matrix, self.is_radial, self.radial_FOV_oversampling_node]
        )
        
        self.reverse_linear_order = ComputeNode(lambda: False, []) # TODO: implement logic (pick forward or reverse order that minimizes readtrin_spacing while respecting minimum spacing and TR)

        self.min_readtrain_spacing = ComputeNode(
            self.min_readtrain_spacing_func,
            [self.is_gradient_echo, self.RF_excitation, self.gre_echo_train_dur, self.readout_risetime, self.read_prephaser_floating, self.phaser_duration, self.slice_select_excitation, self.slice_select_rephaser, self.RF_refocusing_floating, self.slice_select_refocusing_floating]
        )

        self.centermost_rf_echo = ComputeNode(
            self.centermost_rf_echo_func,
            [self.EPI_factor_node, self.is_gradient_echo, self.TE_node, self.min_readtrain_spacing, self.split_center, self.turbo_factor_node]
        )

        self.readtrain_spacing = ComputeNode(
            self.readtrain_spacing_func,
            [self.EPI_factor_node, self.readtrain_spacing_linear_order, self.TE_node, self.centermost_rf_echo, self.split_center]
        )

        self.num_blades = ComputeNode(
            lambda is_radial, matrix, radial_factor, turbo_factor, EPI_factor:
            int(np.ceil(max(matrix) * radial_factor / turbo_factor / EPI_factor * np.pi / 2)) if is_radial else 1,
            [self.is_radial, self.matrix, self.radial_factor_node, self.turbo_factor_node, self.EPI_factor_node]
        )
        
        self.k_angles = ComputeNode(lambda num_blades: np.linspace(0, np.pi, num_blades, endpoint=False), [self.num_blades])
        self.spoke_angle_node = OutputParamNode(self, 'spoke_angle', lambda k_angles, shot: np.degrees(k_angles[shot-1]), [self.k_angles, self.shot_node])

        self.num_shots_node = OutputParamNode(
            self,
            'num_shots',
            lambda matrix_P, phase_oversampling, partial_Fourier, turbo_factor, EPI_factor, is_radial, num_blades:
            int(np.ceil(matrix_P * (1 + phase_oversampling / 100) * partial_Fourier / turbo_factor / EPI_factor)) if not is_radial else num_blades,
            [self.matrix_P_node, self.phase_oversampling_node, self.partial_Fourier_node, self.turbo_factor_node, self.EPI_factor_node, self.is_radial, self.num_blades]
        )

        self.num_measured_lines = ComputeNode(
            lambda turbo_factor, EPI_factor, num_shots, is_radial:
            turbo_factor * EPI_factor * (num_shots if not is_radial else 1), # measured lines per blade
            [self.turbo_factor_node, self.EPI_factor_node, self.num_shots_node, self.is_radial]
        )

        self.k_phase_axis = ComputeNode(
            self.k_phase_axis_func,
            [self.is_radial, self.num_measured_lines, self.matrix, self.phase_dir, self.phase_oversampling_node, self.FOV]
        )

        self.num_blank_lines = ComputeNode(
            lambda k_phase_axis, lines_to_measure:
            len(k_phase_axis) - sum(lines_to_measure),
            [self.k_phase_axis, self.lines_to_measure]
        )

        ActionNode(
            self.set_homodyne_visibility,
            [self.num_blank_lines, self.is_radial]
        )

        self.lines_to_measure = ComputeNode(
            self.lines_to_measure_func,
            [self.k_phase_axis, self.num_measured_lines]
        )

        self.num_segm = ComputeNode(
            lambda num_measured_lines, num_blades, num_shots:
            int(num_measured_lines * num_blades / num_shots),
            [self.num_measured_lines, self.num_blades, self.num_shots_node]
        )
        
        self.num_sym_lines = ComputeNode(
            lambda num_measured_lines, k_phase_axis:
            2 * num_measured_lines - len(k_phase_axis),
            [self.num_measured_lines, self.k_phase_axis]
        )
        
        self.split_center = ComputeNode(
            # does center of k-space lie between two segments?
            lambda num_sym_lines, num_shots:
            (num_sym_lines % num_shots == 0) and ((num_sym_lines / num_shots) % 2 == 0),
            [self.num_sym_lines, self.num_shots_node]
        )
        
        self.num_sym_segm = ComputeNode(
            self.num_sym_segm_func,
            [self.split_center, self.num_sym_lines, self.num_blades, self.num_shots_node]
        )

        self.central_segments = ComputeNode(
            self.central_segments_func,
            [self.split_center, self.num_segm, self.num_sym_segm]
        )

        self.pe_table = ComputeNode(
            self.pe_table_func,
            [self.EPI_factor_node, self.turbo_factor_node, self.num_sym_segm, self.centermost_rf_echo, self.is_radial, self.num_shots_node, self.reverse_linear_order, self.lines_to_measure]
        )

        ActionNode(
            self.set_shot_bounds,
            [self.num_shots]
        )

        self.signal_level = ComputeNode(
            self.signal_level_func,
            [self.k_read_axis, self.lines_to_measure, self.num_blades, self.slice_thickness_node, self.FOV, self.matrix]
        )

        self.spin_echoes = ComputeNode(
            self.spin_echoes_func,
            [self.lines_to_measure, self.pe_table, self.readtrain_spacing]
        )
        
        self.sampling_time = ComputeNode(
            self.sampling_time_func,
            [self.pixel_bandwidth_node, self.k_read_axis]
        )
        
        self.time_after_excitation = ComputeNode(
            self.time_after_excitation_func,
            [self.lines_to_measure, self.pe_table, self.readouts, self.sampling_time, self.freq_dir, self.phase_dir]
        )

        self.time_relative_inphase = ComputeNode(
            self.time_relative_inphase_func,
            [self.time_after_excitation, self.is_gradient_echo, self.spin_echoes, self.phase_dir]
        )
        
        self.dephasing = ComputeNode(
            self.dephasing_func,
            [self.field_strength_node, self.time_relative_inphase]
        )
        
        self.T2w = ComputeNode(
            self.T2w_func,
            [self.tissues, self.time_after_excitation, self.time_relative_inphase, self.field_strength_node]
        )

        self.k_axes = ComputeNode(
            self.k_axes_func,
            [self.freq_dir, self.phase_dir, self.k_read_axis, self.k_phase_axis, self.lines_to_measure]
        )
        
        self.k_grid_axes = ComputeNode(
            self.k_grid_axes_func,
            [self.is_radial, self.k_axes, self.FOV, self.matrix]
        )

        self.k_samples = ComputeNode(
            self.k_samples_func,
            [self.k_axes, self.k_angles]
        )

        self.plain_kspace_comps = ComputeNode(
            self.plain_kspace_comps_func,
            [self.is_radial, self.phantom, self.k_grid_axes, self.k_samples]
        )
        
        self.thick_kspace_comps = ComputeNode(
            self.thick_kspace_comps_func,
            [self.slice_thickness_node, self.k_samples, self.plain_kspace_comps]
        )

        self.kspace_comps = ComputeNode(
            self.kspace_comps_func,
            [self.tissues, self.thick_kspace_comps, self.T2w, self.dephasing]
        )
        
        self.decayed_signal = ComputeNode(
            self.decayed_signal_func,
            [self.signal_level, self.T2w, self.reference_tissue_node, self.k_read_axis, self.k_phase_axis, self.freq_dir]
        )
        
        self.PD_and_T1w = ComputeNode(
            self.PD_and_T1w_func,
            [self.sequence_node, self.TR_node, self.TE_node, self.TI_node, self.FA_node, self.field_strength_node, self.tissues]
        )

        self.reference_signal = ComputeNode(
            lambda decayed_signal, PD_and_T1w, reference_tissue:
            decayed_signal * np.abs(PD_and_T1w[reference_tissue]),
            [self.decayed_signal, self.PD_and_T1w, self.reference_tissue_node]
        )

        self.noise_std = ComputeNode(
            self.noise_std_func,
            [self.sampling_time, self.noise_gain_node, self.NSA_node, self.field_strength_node]
        )

        self.noise = ComputeNode(
            self.noise_func,
            [self.k_samples, self.noise_std]
        )

        self.SNR = ComputeNode(
            lambda signal, noise_std:
            signal / noise_std,
            [self.reference_signal, self.noise_std]
        )
        
        self.relative_SNR_node = OutputParamNode(
            self,
            'relative_SNR',
            lambda SNR, reference_SNR:
            SNR / reference_SNR * 100,
            [self.SNR, self.reference_SNR_node]
        )

        self.FOV_box = ComputeNode(
            self.FOV_box_func,
            [self.show_FOV_node, self.is_radial, self.FOV, self.matrix, self.freq_dir, self.phase_dir, self.k_read_axis, self.k_phase_axis]
        )

        self.scantime_node = OutputParamNode(
            self,
            'scantime',
            lambda num_shots, NSA, TR:
            format_scantime(num_shots * NSA * TR),
            [self.num_shots_node * self.NSA_node * self.TR_node]
        )
        
        self.measured_kspace = ComputeNode(
            self.measured_kspace_func,
            [self.noise, self.kspace_comps, self.FatSat_node, self.PD_and_T1w]
        )

        self.gridded_kspace = ComputeNode(
            self.gridded_kspace_func,
            [self.k_grid_axes, self.is_radial, self.measured_kspace, self.k_samples, self.FOV, self.matrix]
        )

        self.full_kspace = ComputeNode(
            self.full_kspace_func,
            [self.num_blank_lines, self.is_radial, self.gridded_kspace, self.phase_dir, self.homodyne_node, self.k_phase_axis]
        )

        self.full_k_matrix = ComputeNode(lambda full_kspace: full_kspace.shape, [self.full_kspace])
        
        self.apodized_kspace = ComputeNode(
            self.apodized_kspace_func,
            [self.full_kspace, self.do_apodize_node, self.apodization_alpha_node]
        )

        self.oversampled_recon_matrix = ComputeNode(
            self.oversampled_recon_matrix_func,
            [self.recon_matrix, self.full_k_matrix, self.matrix]
        )

        self.zerofilled_kspace = ComputeNode(
            self.zerofilled_kspace_func,
            [self.apodized_kspace, self.oversampled_recon_matrix]
        )
        
        self.image_array = ComputeNode(
            self.image_array_func,
            [self.oversampled_recon_matrix, self.full_k_matrix, self.recon_matrix, self.zerofilled_kspace]
        )
        
        self.image = ComputeNode(
            self.image_func,
            [self.image_type_node, self.recon_matrix, self.FOV, self.image_array]
        )

        self.param.watch(lambda _: self._watch_trajectory(), 'trajectory', precedence=1)

        self.derived_params = ['FOV_bandwidth', 'FW_shift', 'SNR', 'name', 'num_shots', 'recon_voxel_F', 'recon_voxel_P', 'reference_SNR', 'relative_SNR', 'scantime', 'spoke_angle', 'voxel_F', 'voxel_P', 'num_shots_label']
        
        self.sequence_pipeline = {f: True for f in [
            'setup_excitation', 
            'setup_refocusing',
            'setup_inversion',
            'setup_FatSat',
            'setup_slice_selection',
            'setup_readouts',
            'setup_phasers',
            'setup_spoiler',
            'place_refocusing',
            'place_inversion',
            'place_FatSat',
            'place_readouts',
            'place_phasers',
            'place_spoiler',
            'update_min_TE',
            'update_min_TR',
            'update_max_TE',
            'update_max_TI',
            'update_BW_bounds',
            'update_matrix_F_bounds',
            'update_matrix_P_bounds',
            'update_FOV_F_bounds', 
            'update_FOV_P_bounds',
            'update_slice_thickness_bounds'
            ]}
        
        self.acquisition_pipeline = {f: True for f in [
            'sample_kspace', 
            'update_sampling_time', 
            'modulate_kspace', 
            'simulate_noise', 
            'update_PD_and_T1w', 
            'compile_kspace'
        ]}

        self.sequence_plot_pipeline = {f: True for f in [
            'render_frequency_board', 
            'render_phase_board', 
            'render_slice_board', 
            'render_RF_board',
            'render_signal_board',
            'render_TR_span',
            'calculate_k_trajectory'
        ]}
        
        self.recon_pipeline = {f: True for f in [
            'partial_Fourier_recon', 
            'apodization', 
            'zerofill',
            'set_reference_SNR'
        ]}

        self._watch_object()
        self._watch_recon_matrix_F()
        self._watch_recon_matrix_P()
        self._watch_matrix_F()
        self._watch_matrix_P()

        self.run_sequence_pipeline()

        self._initialized = True


    def init_bounds(self):
        self.param.object.objects = phantom.get_phantom_names()
        self.param.field_strength.objects=[1.5, 3.0]
        self.param.parameter_style.objects=['Matrix and Pixel BW', 'voxel_size and Fat/water shift', 'Matrix and FOV BW']
        self.param.frequency_direction.objects=constants.DIRECTIONS.keys()
        self.param.trajectory.objects=constants.TRAJECTORIES[:2]
        self.param.radial_factor.bounds=(0.1, 4.)
        self.param.radial_FOV_oversampling.bounds=(1, 2)
        self.param.sequence.objects=constants.SEQUENCES
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

    def run_pipeline(self, pipeline):
        for f in pipeline:
            if pipeline[f]:
                self.__getattribute__(f)()
                pipeline[f] = False

    def run_sequence_pipeline(self):
        if not any(self.sequence_pipeline.values()):
            return
        self.run_pipeline(self.sequence_pipeline)
        self.resolve_conflicts()

    def run_acquisition_pipeline(self):
        if not any(self.acquisition_pipeline.values()):
            return
        self.run_sequence_pipeline()
        self.run_pipeline(self.acquisition_pipeline)

    def run_sequence_plot_pipeline(self):
        if not any(self.sequence_plot_pipeline.values()):
            return
        self.run_sequence_pipeline()
        self.run_acquisition_pipeline()
        self.run_pipeline(self.sequence_plot_pipeline)

    def run_recon_pipeline(self):
        if not any(self.recon_pipeline.values()):
            return
        self.run_sequence_pipeline()
        self.run_acquisition_pipeline()
        self.run_pipeline(self.recon_pipeline)

    def get_params(self):
        return {param: self.__getattribute__(param) for param in self.param.values().keys() if param not in self.derived_params}

    def set_params(self, settings):
        self.init_bounds()
        self.sequence_pipeline = {f: True for f in self.sequence_pipeline.keys()}
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

    @param.depends('object', watch=True)
    def _watch_object(self):
        if hasattr(self, 'phantom') and self.phantom['name']==self.object:
            return
        self.phantom = phantom.load(self.object, self.min_voxel_size)
        add_to_pipeline(self.sequence_pipeline, ['setup_readouts'])
        self.acquisition_pipeline = {f: True for f in self.acquisition_pipeline}
        self.recon_pipeline = {f: True for f in self.recon_pipeline}
        self.tissues = list(self.phantom['shapes'].keys())
        self.param.reference_tissue.objects = self.tissues
        self.reference_tissue = self.tissues[0]
        min_FOV = self.phantom['support']
        if self.frequency_direction=='left-right':
            min_FOV = min_FOV.reverse()
        with param.parameterized.batch_call_watchers(self):
            self.FOV_F = max(self.FOV_F, min_FOV[0])
            self.FOV_P = max(self.FOV_P, min_FOV[1])

    @param.depends('parameter_style', watch=True)
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

    @param.depends('FOV_F', watch=True)
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
            add_to_pipeline(self.acquisition_pipeline, ['sample_kspace', 'update_sampling_time', 'modulate_kspace', 'simulate_noise', 'compile_kspace'])
            add_to_pipeline(self.recon_pipeline, ['partial_Fourier_recon', 'apodization', 'zerofill'])
            add_to_pipeline(self.sequence_pipeline, ['setup_readouts', 'update_BW_bounds', 'update_matrix_F_bounds'])

    @param.depends('FOV_P', watch=True)
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
            add_to_pipeline(self.acquisition_pipeline, ['sample_kspace', 'update_sampling_time', 'modulate_kspace', 'simulate_noise', 'compile_kspace'])
            add_to_pipeline(self.recon_pipeline, ['partial_Fourier_recon', 'apodization', 'zerofill'])
            add_to_pipeline(self.sequence_pipeline, ['setup_phasers', 'update_matrix_P_bounds'])

    @param.depends('phase_oversampling', watch=True)
    def _watch_phase_oversampling(self):
        self._watch_FOV_P()

    @param.depends('radial_factor', watch=True)
    def _watch_radial_factor(self):
        pass

    @param.depends('num_shots', watch=True)
    def _watch_num_shots(self):
        add_to_pipeline(self.acquisition_pipeline, ['sample_kspace', 'modulate_kspace', 'simulate_noise', 'compile_kspace'])
        add_to_pipeline(self.recon_pipeline, ['partial_Fourier_recon', 'apodization', 'zerofill'])
        add_to_pipeline(self.sequence_pipeline, ['setup_phasers', 'update_FOV_P_bounds', 'update_turbo_factor_bounds', 'update_EPI_factor_objects'])
        add_to_pipeline(self.sequence_plot_pipeline, ['calculate_k_trajectory'])

    @param.depends('matrix_F', watch=True)
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
            add_to_pipeline(self.acquisition_pipeline, ['sample_kspace', 'update_sampling_time', 'modulate_kspace', 'simulate_noise', 'compile_kspace'])
            add_to_pipeline(self.recon_pipeline, ['partial_Fourier_recon', 'apodization', 'zerofill'])
            add_to_pipeline(self.sequence_pipeline, ['setup_readouts', 'update_BW_bounds', 'update_matrix_F_bounds', 'update_FOV_F_bounds'])
            self.set_param_bounds(self.param['recon_matrix_F'], minval=self.matrix_F)
            self.update_recon_voxel_F_objects()
            self.set_closest(self.param.recon_matrix_F, self.matrix_F * self.rec_acq_ratio_F)
            self.param.trigger('voxel_F', 'recon_voxel_F')
            if self.is_radial.value:
                self.set_closest(self.param.matrix_P, self.matrix_F * self.FOV_P / self.FOV_F)

    @param.depends('matrix_P', watch=True)
    def _watch_matrix_P(self):
        with param.parameterized.batch_call_watchers(self):
            self.update_voxel_P_objects()
            self.set_closest(self.param.voxel_P, self.FOV_P / self.matrix_P)
            add_to_pipeline(self.acquisition_pipeline, ['sample_kspace', 'update_sampling_time', 'modulate_kspace', 'simulate_noise', 'compile_kspace'])
            add_to_pipeline(self.recon_pipeline, ['partial_Fourier_recon', 'apodization', 'zerofill'])
            add_to_pipeline(self.sequence_pipeline, ['setup_phasers', 'update_FOV_P_bounds', 'update_turbo_factor_bounds', 'update_EPI_factor_objects'])
            self.set_param_bounds(self.param['recon_matrix_P'], minval=self.matrix_P)
            self.update_recon_voxel_P_objects()
            self.set_closest(self.param.recon_matrix_P, self.matrix_P * self.rec_acq_ratio_P)
            self.param.trigger('voxel_P', 'recon_voxel_P')
            if self.is_radial.value:
                self.set_closest(self.param.matrix_F, self.matrix_P * self.FOV_F / self.FOV_P)

    @param.depends('voxel_F', watch=True)
    def _watch_voxel_F(self):
        if self.param.voxel_F.precedence > 0:
            self.set_closest(self.param.matrix_F, self.FOV_F / self.voxel_F)
            add_to_pipeline(self.sequence_pipeline, ['update_FOV_F_bounds'])

    @param.depends('voxel_P', watch=True)
    def _watch_voxel_P(self):
        if self.param.voxel_P.precedence > 0:
            self.set_closest(self.param.matrix_P, self.FOV_P / self.voxel_P)
            add_to_pipeline(self.sequence_pipeline, ['update_FOV_P_bounds'])

    @param.depends('recon_voxel_F', watch=True)
    def _watch_recon_voxel_F(self):
        if self.param.recon_voxel_F.precedence > 0:
            self.set_closest(self.param.recon_matrix_F, self.FOV_F / self.recon_voxel_F)
            add_to_pipeline(self.sequence_pipeline, ['update_FOV_F_bounds'])

    @param.depends('recon_voxel_P', watch=True)
    def _watch_recon_voxel_P(self):
        if self.param.recon_voxel_P.precedence > 0:
            self.set_closest(self.param.recon_matrix_P, self.FOV_P / self.recon_voxel_P)
            add_to_pipeline(self.sequence_pipeline, ['update_FOV_P_bounds'])

    @param.depends('slice_thickness', watch=True)
    def _watch_slice_thickness(self):
        add_to_pipeline(self.acquisition_pipeline, ['sample_kspace', 'update_sampling_time', 'modulate_kspace', 'simulate_noise', 'compile_kspace'])
        add_to_pipeline(self.recon_pipeline, ['partial_Fourier_recon', 'apodization', 'zerofill'])
        add_to_pipeline(self.sequence_pipeline, ['setup_slice_selection', 'place_FatSat'])

    @param.depends('radial_FOV_oversampling', watch=True)
    def _watch_radial_FOV_oversampling(self):
        add_to_pipeline(self.acquisition_pipeline, ['sample_kspace', 'update_sampling_time', 'modulate_kspace', 'simulate_noise', 'compile_kspace'])
        add_to_pipeline(self.recon_pipeline, ['partial_Fourier_recon', 'apodization', 'zerofill'])

    @param.depends('frequency_direction', watch=True)
    def _watch_frequency_direction(self):
        add_to_pipeline(self.acquisition_pipeline, ['sample_kspace', 'update_sampling_time', 'modulate_kspace', 'simulate_noise', 'compile_kspace'])
        add_to_pipeline(self.recon_pipeline, ['partial_Fourier_recon', 'apodization', 'zerofill'])
        for p in [self.param.FOV_F, self.param.FOV_P, self.param.matrix_F, self.param.matrix_P, self.param.recon_matrix_F, self.param.recon_matrix_P]:
            if ' x' in p.label:
                p.label = p.label.replace(' x', ' y')
            elif ' y' in p.label:
                p.label = p.label.replace(' y', ' x')
         # frequency oversampling is adapted to phantom FOV for efficiency
        add_to_pipeline(self.sequence_plot_pipeline, ['calculate_k_trajectory'])

    #@param.depends('trajectory', watch=True)
    def _watch_trajectory(self):
        add_to_pipeline(self.acquisition_pipeline, ['sample_kspace', 'update_sampling_time', 'modulate_kspace', 'simulate_noise', 'compile_kspace'])
        add_to_pipeline(self.recon_pipeline, ['partial_Fourier_recon', 'apodization', 'zerofill'])
        add_to_pipeline(self.sequence_pipeline, ['setup_readouts', 'setup_phasers', 'update_FOV_P_bounds', 'update_turbo_factor_bounds', 'update_EPI_factor_objects'])
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
        
        add_to_pipeline(self.sequence_plot_pipeline, ['calculate_k_trajectory'])

    @param.depends('field_strength', watch=True)
    def _watch_field_strength(self):
        with param.parameterized.batch_call_watchers(self):
            self.update_FW_shift_objects()
            self.set_closest(self.param.FW_shift, pixel_BW_to_shift(self.pixel_bandwidth, self.field_strength))
            self.param.trigger('FW_shift')
            add_to_pipeline(self.acquisition_pipeline, ['update_sampling_time', 'modulate_kspace', 'simulate_noise', 'update_PD_and_T1w', 'compile_kspace'])
            add_to_pipeline(self.recon_pipeline, ['partial_Fourier_recon', 'apodization', 'zerofill'])
            self._watch_FatSat() # since fatsat pulse duration depends on field_strength

    @param.depends('pixel_bandwidth', watch=True)
    def _watch_pixel_bandwidth(self):
        with param.parameterized.batch_call_watchers(self):
            self.set_closest(self.param.FW_shift, pixel_BW_to_shift(self.pixel_bandwidth, self.field_strength))
            self.set_closest(self.param.FOV_bandwidth, pixel_BW_to_FOV_BW(self.pixel_bandwidth, self.matrix_F))
            add_to_pipeline(self.acquisition_pipeline, ['update_sampling_time', 'modulate_kspace', 'simulate_noise', 'compile_kspace'])
            add_to_pipeline(self.recon_pipeline, ['partial_Fourier_recon', 'apodization', 'zerofill'])
            add_to_pipeline(self.sequence_pipeline, ['setup_readouts', 'update_matrix_F_bounds', 'update_FOV_F_bounds', 'update_matrix_P_bounds', 'update_FOV_P_bounds'])

    @param.depends('FW_shift', watch=True)
    def _watch_FW_shift(self):
        if self.param.FW_shift.precedence > 0:
            self.set_closest(self.param.pixel_bandwidth, shift_to_pixel_BW(self.FW_shift, self.field_strength))

    @param.depends('FOV_bandwidth', watch=True)
    def _watch_FOV_bandwidth(self):
        if self.param.FOV_bandwidth.precedence > 0:
            self.set_closest(self.param.pixel_bandwidth, FOV_BW_to_pixel_BW(self.FOV_bandwidth, self.matrix_F))

    @param.depends('NSA', watch=True)
    def _watch_NSA(self):
        add_to_pipeline(self.acquisition_pipeline, ['update_sampling_time', 'modulate_kspace', 'simulate_noise', 'compile_kspace'])
        add_to_pipeline(self.recon_pipeline, ['partial_Fourier_recon', 'apodization', 'zerofill'])

    @param.depends('partial_Fourier', watch=True)
    def _watch_partial_Fourier(self):
        add_to_pipeline(self.acquisition_pipeline, ['sample_kspace', 'update_sampling_time', 'modulate_kspace', 'simulate_noise', 'compile_kspace'])
        add_to_pipeline(self.recon_pipeline, ['partial_Fourier_recon', 'apodization', 'zerofill'])
        add_to_pipeline(self.sequence_pipeline, ['setup_refocusing', 'setup_readouts', 'setup_phasers', 'update_min_TE', 'update_min_TR', 'update_max_TE', 'update_max_TI', 'update_BW_bounds', 'update_matrix_F_bounds', 'update_matrix_P_bounds', 'update_FOV_F_bounds',  'update_FOV_P_bounds', 'update_slice_thickness_bounds', 'update_turbo_factor_bounds', 'update_EPI_factor_objects'])

    @param.depends('turbo_factor', watch=True)
    def _watch_turbo_factor(self):
        add_to_pipeline(self.acquisition_pipeline, ['sample_kspace', 'update_sampling_time', 'modulate_kspace', 'simulate_noise', 'compile_kspace'])
        add_to_pipeline(self.recon_pipeline, ['partial_Fourier_recon', 'apodization', 'zerofill'])
        add_to_pipeline(self.sequence_pipeline, ['setup_refocusing', 'setup_readouts', 'setup_phasers', 'update_min_TE', 'update_min_TR', 'update_max_TE', 'update_max_TI', 'update_BW_bounds', 'update_matrix_F_bounds', 'update_matrix_P_bounds', 'update_FOV_F_bounds',  'update_FOV_P_bounds', 'update_slice_thickness_bounds', 'update_EPI_factor_objects'])
        self.update_labels_by_trajectory()
        self.update_EPI_factor_objects()

    @param.depends('EPI_factor', watch=True)
    def _watch_EPI_factor(self):
        add_to_pipeline(self.acquisition_pipeline, ['sample_kspace', 'update_sampling_time', 'modulate_kspace', 'simulate_noise', 'compile_kspace'])
        add_to_pipeline(self.recon_pipeline, ['partial_Fourier_recon', 'apodization', 'zerofill'])
        add_to_pipeline(self.sequence_pipeline, ['setup_readouts', 'setup_phasers', 'update_min_TE', 'update_min_TR', 'update_max_TE', 'update_max_TI', 'update_BW_bounds', 'update_matrix_F_bounds', 'update_matrix_P_bounds', 'update_FOV_F_bounds',  'update_FOV_P_bounds', 'update_slice_thickness_bounds', 'update_turbo_factor_bounds'])
        self.update_labels_by_trajectory()
        self.update_turbo_factor_bounds()

    @param.depends('shot', watch=True)
    def _watch_shot(self):
        add_to_pipeline(self.sequence_pipeline, ['setup_phasers'])
        add_to_pipeline(self.sequence_plot_pipeline, ['render_signal_board', 'calculate_k_trajectory'])

    @param.depends('sequence', watch=True)
    def _watch_sequence(self):
        add_to_pipeline(self.acquisition_pipeline, ['modulate_kspace', 'update_PD_and_T1w', 'compile_kspace'])
        add_to_pipeline(self.recon_pipeline, ['partial_Fourier_recon', 'apodization', 'zerofill'])
        add_to_pipeline(self.sequence_pipeline, ['setup_excitation', 'setup_refocusing', 'setup_inversion', 'setup_phasers', 'place_readouts', 'place_phasers'])
        self.param.FA.precedence = 1 if self.sequence=='Spoiled Gradient Echo' else -1
        self.param.TI.precedence = 1 if self.sequence=='Inversion Recovery' else -1
        if self.sequence=='Spoiled Gradient Echo':
            self.turbo_factor = 1
            self.param.turbo_factor.precedence = -6
        else:
            self.param.turbo_factor.precedence = 6

    @param.depends('TE', watch=True)
    def _watch_TE(self):
        add_to_pipeline(self.acquisition_pipeline, ['modulate_kspace', 'update_PD_and_T1w', 'compile_kspace'])
        add_to_pipeline(self.recon_pipeline, ['partial_Fourier_recon', 'apodization', 'zerofill'])
        add_to_pipeline(self.sequence_pipeline, ['setup_phasers', 'place_refocusing', 'place_readouts', 'update_matrix_F_bounds', 'update_FOV_F_bounds', 'update_matrix_P_bounds', 'update_FOV_P_bounds'])

    @param.depends('TR', watch=True)
    def _watch_TR(self):
        add_to_pipeline(self.acquisition_pipeline, ['update_PD_and_T1w', 'compile_kspace'])
        add_to_pipeline(self.recon_pipeline, ['partial_Fourier_recon', 'apodization', 'zerofill'])
        add_to_pipeline(self.sequence_pipeline, ['update_max_TE', 'update_max_TI', 'update_BW_bounds', 'update_slice_thickness_bounds'])
        add_to_pipeline(self.sequence_plot_pipeline, ['render_TR_span'])

    @param.depends('TI', watch=True)
    def _watch_TI(self):
        add_to_pipeline(self.acquisition_pipeline, ['update_PD_and_T1w', 'compile_kspace'])
        add_to_pipeline(self.recon_pipeline, ['partial_Fourier_recon', 'apodization', 'zerofill'])
        add_to_pipeline(self.sequence_pipeline, ['place_inversion'])

    @param.depends('FA', watch=True)
    def _watch_FA(self):
        add_to_pipeline(self.acquisition_pipeline, ['update_PD_and_T1w', 'compile_kspace'])
        add_to_pipeline(self.recon_pipeline, ['partial_Fourier_recon', 'apodization', 'zerofill'])
        add_to_pipeline(self.sequence_pipeline, ['setup_excitation'])

    @param.depends('FatSat', watch=True)
    def _watch_FatSat(self):
        add_to_pipeline(self.acquisition_pipeline, ['compile_kspace'])
        add_to_pipeline(self.recon_pipeline, ['partial_Fourier_recon', 'apodization', 'zerofill'])
        add_to_pipeline(self.sequence_pipeline, ['setup_FatSat', 'update_max_TE', 'update_BW_bounds'])

    @param.depends('homodyne', watch=True)
    def _watch_homodyne(self):
        add_to_pipeline(self.recon_pipeline, ['partial_Fourier_recon', 'apodization', 'zerofill'])

    @param.depends('do_apodize', watch=True)
    def _watch_do_apodize(self):
        if self.do_apodize:
            self.param.apodization_alpha.precedence = abs(self.param.apodization_alpha.precedence)
        else:
            self.param.apodization_alpha.precedence = -abs(self.param.apodization_alpha.precedence)
        add_to_pipeline(self.recon_pipeline, ['apodization', 'zerofill'])

    @param.depends('apodization_alpha', watch=True)
    def _watch_apodization_alpha(self):
        add_to_pipeline(self.recon_pipeline, ['apodization', 'zerofill'])

    @param.depends('do_zerofill', watch=True)
    def _watch_do_zerofill(self):
        add_to_pipeline(self.recon_pipeline, ['zerofill'])

    @param.depends('recon_matrix_F', watch=True)
    def _watch_recon_matrix_F(self):
        self.rec_acq_ratio_F = self.recon_matrix_F / self.matrix_F
        if self.do_zerofill:
            add_to_pipeline(self.recon_pipeline, ['zerofill'])
        self.set_closest(self.param.recon_voxel_F, self.FOV_F / self.recon_matrix_F)

    @param.depends('recon_matrix_P', watch=True)
    def _watch_recon_matrix_P(self):
        self.rec_acq_ratio_P = self.recon_matrix_P / self.matrix_P
        if self.do_zerofill:
            add_to_pipeline(self.recon_pipeline, ['zerofill'])
        self.set_closest(self.param.recon_voxel_P, self.FOV_P / self.recon_matrix_P)

    @param.depends('reference_tissue', watch=True)
    def _watch_reference_tissue(self):
        add_to_pipeline(self.acquisition_pipeline, ['modulate_kspace', 'compile_kspace'])
        if self._initialized:
            self.run_acquisition_pipeline()

    def get_sequence_start(self):
        if self.sequence == 'Inversion Recovery': 
            return self.boards['slice']['objects']['slice select inversion']['time'][0]
        elif self.FatSat:
            return self.boards['RF']['objects']['fatsat']['time'][0]
        else:
            return self.boards['slice']['objects']['slice select excitation']['time'][0]

    def get_TE_from_centermost_echoes(self, readtrain_spacing, centermost_gr_echoes, centermost_rf_echoes):
        TE = readtrain_spacing * (1 + np.mean(centermost_rf_echoes))
        readtrain_shift = self.gr_echo_spacing * (np.mean(centermost_gr_echoes) - (self.EPI_factor-1)/2)
        TE += readtrain_shift
        return TE

    def update_min_TE(self):
        min_readtrain_spacing = self.get_min_readtrain_spacing()
        if self.EPI_factor == 1: # flexible segment order for (turbo) spin echo
            min_centermost_gr_echoes = [0]
            min_centermost_rf_echoes = [0]
            if (self.split_center and self.turbo_factor > 1):
                min_centermost_rf_echoes += [1]
            self.min_TE = self.get_TE_from_centermost_echoes(min_readtrain_spacing, min_centermost_gr_echoes, min_centermost_rf_echoes)
        else: # linear segment order for EPI and GRASE
            # # pick forward or reverse order that minimizes TE (may be forward for GRASE)
            TE_cands = [self.get_TE_from_centermost_echoes(min_readtrain_spacing, *self.get_centermost_echoes_linear_order(reverse)) for reverse in [True, False]]
            self.min_TE = min(TE_cands)
        add_to_pipeline(self.sequence_pipeline, ['update_max_TE'])

    def update_min_TR(self):
        self.min_TR = self.boards['slice']['objects']['spoiler']['time'][-1]
        self.min_TR -= self.get_sequence_start()
        self.set_param_bounds(self.param.TR, minval=self.min_TR)
        add_to_pipeline(self.sequence_pipeline, ['update_max_TE', 'update_max_TI'])

    def update_max_TE(self):
        max_TE = self.TR - self.min_TR + self.TE
        self.set_param_bounds(self.param.TE, minval=self.min_TE, maxval=max_TE)

    def update_max_TI(self):
        if self.sequence != 'Inversion Recovery':
            return
        max_TI = self.TR - self.min_TR + self.TI
        self.set_param_bounds(self.param.TI, maxval=max_TI)

    def resolve_conflicts(self, max_TR=False):
        if max_TR:
            self.outbound_params.add('TR')
        if self.outbound_params:
            for par in ['TR', 'TI', 'TE', 'pixel_bandwidth']:
                if par in self.outbound_params:
                    value = getattr(self, par)
                    if par=='TR' and max_TR:
                        warnings.warn('Resolving conflict by maximizing TR')
                        add_to_pipeline(self.sequence_pipeline, ['update_min_TR'])
                        self.TR = list(param_values['TR'].values())[-1]
                    else:
                        warnings.warn(f'Resolving conflict: {par}')
                    self.set_closest(self.param[par], value) # Set back param within bounds
        if self.outbound_params:
            if not max_TR:
                self.resolve_conflicts(max_TR=True)
            else:
                warnings.warn(f'Unresolved conflict: {self.outbound_params}')

    def get_max_prephaser_area(self, readAmp):
        if self.is_gradient_echo.value:
            max_prephaser_dur =  self.TE - self.boards['ADC']['objects']['samplings'][0][0]['dur_f']/2 - self.boards['RF']['objects']['excitation']['dur_f']/2 - readAmp/self.max_slew
        else:
            max_prephaser_dur =  self.TE/2 - self.boards['RF']['objects']['refocusing'][0]['dur_f']/2 - self.boards['RF']['objects']['excitation']['dur_f']/2
        max_prephaser_flat_dur = max_prephaser_dur - (2 * self.max_amp/self.max_slew)
        if max_prephaser_flat_dur < 0: # triangle
            max_prephaser_area = max_prephaser_dur**2 * self.max_slew / 4
        else: # trapezoid
            slewArea = self.max_amp**2 / self.max_slew
            flat_area = self.max_amp * max_prephaser_flat_dur
            max_prephaser_area = slewArea + flat_area
        return max_prephaser_area

    def get_max_readout_area(self):
        max_readout_areas = []
        # See paramBounds.tex for formulae
        d = 1e3 / self.pixel_bandwidth # readout duration
        s = self.max_slew
        if self.is_gradient_echo.value:
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
                    q = t - max(self.phaser_duration,
                                self.boards['slice']['objects']['slice select excitation']['risetime_f'] + self.boards['slice']['objects']['slice select rephaser']['dur_f'])
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
                v = max(self.max_blip_dur - 2 * read_risetime, 0)
        else: # (turbo) spin echo / GRASE
            # limit by half readout duration tr:
            tr = (self.readtrain_spacing - self.boards['RF']['objects']['refocusing'][0]['dur_f']) / self.EPI_factor / 2
            Ar = d*s* tr - d**2*s/2
            max_readout_areas.append(Ar)
            # limit by prephaser duration tp:
            tp = (self.readtrain_spacing - self.boards['RF']['objects']['refocusing'][0]['dur_f'] - self.boards['RF']['objects']['excitation']['dur_f'])/2
            h = s * tp / 2
            h = min(h, self.max_amp)
            Ap = d * (np.sqrt((d*s)**2 - 8*h*(h-s*tp)) - d*s) / 2
            max_readout_areas.append(Ap)
        max_readout_areas.append(self.max_amp * 1e3 / self.pixel_bandwidth) # max wrt max_amp
        return min(max_readout_areas)

    def get_max_phaser_area(self):
        if self.is_gradient_echo.value:
            max_phaser_duration = self.readtrain_spacing - self.boards['RF']['objects']['excitation']['dur_f']/2 - self.gre_echo_train_dur/2 + self.readout_risetime
        else:
            max_phaser_duration = (self.readtrain_spacing - self.boards['RF']['objects']['refocusing'][0]['dur_f'] - self.gre_echo_train_dur)/2 + self.readout_risetime
        max_risetime = self.max_amp / self.max_slew
        if max_phaser_duration > 2 * max_risetime: # trapezoid maxPhaser
            max_phaserarea = (max_phaser_duration - max_risetime) * self.max_amp
        else: # triangular maxPhaser
            max_phaserarea = (max_phaser_duration/2)**2 * self.max_slew
        return max_phaserarea

    def get_max_read_duration(self, read_start_to_kcenter, kcenter_to_read_end, reverse=False):
        # simplification: use current risetime (self.readout_risetime)
        centermost_gr_echoes, centermost_rf_echoes = self.get_centermost_echoes_linear_order(reverse=reverse)
        if len(centermost_gr_echoes)==1:
            num_early_readouts = centermost_gr_echoes[0] + 1/2
            num_early_ramps = centermost_gr_echoes[0] * 2
        else:
            num_early_readouts = max(centermost_gr_echoes)
            num_early_ramps = num_early_readouts * 2 - 1
        # max limit imposed by TE:
        max_read_dur_early = ((read_start_to_kcenter - num_early_ramps * self.readout_risetime) / num_early_readouts)
        num_late_readouts = self.EPI_factor - num_early_readouts
        num_late_ramps = (self.EPI_factor - 1) * 2 - num_early_ramps
        # max limit imposed by TR:
        max_read_dur_late = ((kcenter_to_read_end - num_late_ramps * self.readout_risetime) / num_late_readouts)
        return min(max_read_dur_early, max_read_dur_late)

    def update_labels_by_trajectory(self):
        shot_label = 'shot' if not self.is_radial.value else 'spoke' if (self.EPI_factor * self.turbo_factor == 1) else 'blade'
        self.num_shots_label = f'# {shot_label}s'
        self.param.shot.label = f'Displayed {shot_label}'
        self.param.radial_factor.label = f'{shot_label.capitalize()} sampling factor'
        # Label radial trajectory 'Radial' or 'PROPELLER' depending on nLines per shot
        traj_indices = [0, 1] if (self.EPI_factor * self.turbo_factor == 1) else [0, 2]
        self.param.trajectory.objects = [constants.TRAJECTORIES[i] for i in traj_indices]
        if self.trajectory not in self.param.trajectory.objects:
            self.trajectory = constants.TRAJECTORIES[traj_indices[-1]]

    def update_BW_bounds(self):
        # See paramBounds.tex for formulae relating to the readout board
        s = self.max_slew
        A = 1e3 * self.matrix_F / (self.FOV_F * constants.GYRO) # readout area
        # min limit imposed by maximum gradient amplitude:
        min_read_durations = [A / self.max_amp]
        max_read_durations = []
        if self.is_gradient_echo.value:
            min_phaser_time = min([self.boards['phase']['objects'][typ][0]['dur_f'] for typ in ['phasers', 'rephasers']])
            read_start_to_TE = self.TE - self.boards['RF']['objects']['excitation']['dur_f']/2
            read_start_to_TE -= max(
                self.boards['frequency']['objects']['read prephaser']['dur_f'] + self.readout_risetime, # TODO: consider maximum dur+risetime, not only current (difficult!)
                min_phaser_time,
                self.boards['slice']['objects']['slice select excitation']['risetime_f'] + self.boards['slice']['objects']['slice select rephaser']['dur_f'])
            TE_to_spoiler = (self.TR - (-self.get_sequence_start()) - self.boards['slice']['objects']['spoiler']['dur_f']) - self.TE
            # pick forward or reverse order that maximizes read duration limit
            max_read_durs = []
            for reverse in [True, False]:
                max_read_durs.append(self.get_max_read_duration(read_start_to_TE, TE_to_spoiler, reverse))
            max_read_durations.append(max(max_read_durs))
        else: # spin echo
            refocusing_dur = self.boards['RF']['objects']['refocusing'][0]['dur_f']
            if self.turbo_factor==1 and self.EPI_factor==1: # prephaser should only be limiting for pure spin echo
                # min limit imposed by prephaser duration tp:
                tp = self.TE/2 - refocusing_dur/2 - self.boards['RF']['objects']['excitation']['dur_f']/2
                h = s * tp / 2
                h = min(h, self.max_amp)
                denom = 2*h*s*tp - s*A - 2*h**2
                min_read_durations.append(np.sqrt(A**2/denom) if denom > 0 else np.inf)
            if self.EPI_factor==1:
                max_readtrain_spacing = self.TE / (1 + 1/2 * self.split_center)
            else: # linear k-space order
                # TODO: correct this
                max_readtrain_spacing = max([self.get_readtrain_spacing_linear_order(reverse) for reverse in [True, False]])
            idle_space = max_readtrain_spacing - self.boards['RF']['objects']['refocusing'][0]['dur_f']
            # max limit imposed by phaser:
            max_read_durations.append((idle_space - 2 * self.phaser_duration - self.max_blip_dur * (self.EPI_factor-1))/self.EPI_factor)
            # tr is half the maximum readout gradient duration
            tr = ((idle_space) / self.EPI_factor) / 2
            # max limit imposed by readout rise time:
            radicand = tr**2 - 2*A/s
            if radicand >= 0:
                max_read_durations.append(tr + np.sqrt(radicand))
            else:
                max_read_durations.append(0)
            # max limit imposed by slice select refocusing down ramp time:
            max_read_durations.append((tr - self.boards['slice']['objects']['slice select refocusing'][0]['risetime_f']) * 2)
            # readtrain_spacing may be limited by TR:
            read_end_by_TR = (self.TR - (-self.get_sequence_start()) - self.boards['slice']['objects']['spoiler']['dur_f'])
            read_end_by_else = self.readtrain_spacing * (self.turbo_factor + 1/2) - refocusing_dur/2
            max_read_durations.append((tr - (read_end_by_else-read_end_by_TR)) * 2)
        min_read_duration, max_read_duration = max(min_read_durations), min(max_read_durations)
        small = 1e-2 # to avoid roundoff errors
        min_pixel_BW = 1e3 / max_read_duration + small if max_read_duration > 0 else np.inf
        max_pixel_BW = 1e3 / min_read_duration - small if min_read_duration > 0 else np.inf
        self.set_param_bounds(self.param.pixel_bandwidth, minval=min_pixel_BW, maxval=max_pixel_BW)
        self.update_FW_shift_objects()
        self.update_FOV_bandwidth_objects()

    def update_matrix_F_bounds(self):
        min_matrix_F = []
        max_matrix_F = [self.get_max_readout_area() * 1e-3 * self.FOV_F * constants.GYRO]
        if self.parameter_style == 'Matrix and FOV BW':
            min_matrix_F.append(self.pixel_bandwidth * self.matrix_F / list(self.param.pixel_bandwidth.objects.values())[-1])
            max_matrix_F.append(self.pixel_bandwidth * self.matrix_F / list(self.param.pixel_bandwidth.objects.values())[0])
        self.set_param_bounds(self.param.matrix_F, minval=min_matrix_F, maxval=max_matrix_F)
        self.update_voxel_F_objects()
        self.update_recon_voxel_F_objects()

    def update_matrix_P_bounds(self):
        max_matrix_P = int(self.get_max_phaser_area() * 2e-3 * self.FOV_P * constants.GYRO) + 1
        self.set_param_bounds(self.param.matrix_P, maxval=max_matrix_P)
        self.update_voxel_P_objects()
        self.update_recon_voxel_P_objects()

    def update_FOV_F_bounds(self):
        max_readout_area = self.get_max_readout_area()
        min_FOV = [1e3 * self.matrix_F / (max_readout_area * constants.GYRO) if max_readout_area > 0 else np.inf]
        max_FOV = []
        if self.parameter_style == 'voxel_size and Fat/water shift':
            min_FOV.append(self.FOV_F / self.matrix_F * self.param.matrix_F.objects[0])
            min_FOV.append(self.FOV_F / self.recon_matrix_F * self.param.recon_matrix_F.objects[0])
            max_FOV.append(self.FOV_F / self.matrix_F * self.param.matrix_F.objects[-1])
            max_FOV.append(self.FOV_F / self.recon_matrix_F * self.param.recon_matrix_F.objects[-1])
        self.set_param_bounds(self.param.FOV_F, minval=min_FOV, maxval=max_FOV)

    def update_FOV_P_bounds(self):
        min_FOV = [(self.matrix_P - 1) / (self.get_max_phaser_area() * constants.GYRO * 2e-3)]
        max_FOV = []
        if self.parameter_style == 'voxel_size and Fat/water shift':
            min_FOV.append(self.FOV_P / self.matrix_P * self.param.matrix_P.objects[0])
            min_FOV.append(self.FOV_P / self.recon_matrix_P * self.param.recon_matrix_P.objects[0])
            max_FOV.append(self.FOV_P / self.matrix_P * self.param.matrix_P.objects[-1])
            max_FOV.append(self.FOV_P / self.recon_matrix_P * self.param.recon_matrix_P.objects[-1])
        self.set_param_bounds(self.param.FOV_P, minval=min_FOV, maxval=max_FOV)

    def update_FW_shift_objects(self):
        self.param.FW_shift.objects = {f'{format_float(shift, 2)} pixels': shift for shift in [pixel_BW_to_shift(pBW, self.field_strength) for pBW in list(self.param.pixel_bandwidth.objects.values())[::-1]]}

    def update_FOV_bandwidth_objects(self):
        self.param.FOV_bandwidth.objects = {f'±{format_float(bw, 3)} kHz': bw for bw in [pixel_BW_to_FOV_BW(pBW, self.matrix_F) for pBW in self.param.pixel_bandwidth.objects.values()]}

    def update_voxel_F_objects(self):
        self.param.voxel_F.objects = {f'{format_float(voxel, 3)} mm': voxel for voxel in [self.FOV_F / matrix for matrix in self.param.matrix_F.objects[::-1]]}

    def update_voxel_P_objects(self):
        self.param.voxel_P.objects = {f'{format_float(voxel, 3)} mm': voxel for voxel in [self.FOV_P / matrix for matrix in self.param.matrix_P.objects[::-1]]}

    def update_recon_voxel_F_objects(self):
        self.param.recon_voxel_F.objects = {f'{format_float(voxel, 3)} mm': voxel for voxel in [self.FOV_F / matrix for matrix in self.param.recon_matrix_F.objects[::-1]]}

    def update_recon_voxel_P_objects(self):
        self.param.recon_voxel_P.objects = {f'{format_float(voxel, 3)} mm': voxel for voxel in [self.FOV_P / matrix for matrix in self.param.recon_matrix_P.objects[::-1]]}

    def update_slice_thickness_bounds(self):
        min_thks = [self.boards['RF']['objects']['excitation']['FWHM_f'] / (self.max_amp * constants.GYRO)]
        if not self.is_gradient_echo.value:
            min_thks.append(self.boards['RF']['objects']['refocusing'][0]['FWHM_f'] / (self.max_amp * constants.GYRO))
        if self.sequence=='Inversion Recovery':
            min_thks.append(self.boards['RF']['objects']['inversion']['FWHM_f'] / (self.max_amp * constants.GYRO) * self.inversion_thk_factor)
        
        # Constraint due to TR: 
        if self.sequence=='Inversion Recovery':
            max_risetime = self.TR - (self.boards['slice']['objects']['spoiler']['time'][-1] - self.boards['RF']['objects']['inversion']['time'][0])
            max_amp = self.max_slew * max_risetime
            min_thks.append(self.boards['RF']['objects']['inversion']['FWHM_f'] / (max_amp * constants.GYRO))
        else:
            max_risetime = self.TR - (self.boards['slice']['objects']['spoiler']['time'][-1] - self.boards['RF']['objects']['excitation']['time'][0])
            max_amp = self.max_slew * max_risetime
            min_thks.append(self.boards['RF']['objects']['excitation']['FWHM_f'] / (max_amp * constants.GYRO))
        
        # See paramBounds.tex for formulae
        s = self.max_slew
        d = self.boards['RF']['objects']['excitation']['dur_f']
        if self.is_gradient_echo.value: # Constraint due to slice rephaser
            t = self.boards['ADC']['objects']['samplings'][0][0]['time'][0]
            h = s * (t - np.sqrt(t**2/2 + d**2/8))
            h = min(h, self.max_amp)
            A = d * (np.sqrt((d*s+2*h)**2 - 8*h*(h-s*(t-d/2))) - d*s - 2*h) / 2
        else: # Spin echo: Constraint due to slice rephaser and refocusing slice select rampup
            t = self.boards['RF']['objects']['refocusing'][0]['time'][0]
            h = s * (np.sqrt(2*(d + 2*t)**2 - 4*d**2) - d - 2*t) / 4
            h = min(h, self.max_amp)
            A = (np.sqrt((d*(d*s + 4*h))**2 - 4*d**2*h*(d*s + 2*h - 2*s*t)) - d*(d*s + 4*h)) / 2
        Be = self.boards['RF']['objects']['excitation']['FWHM_f']
        min_thks.append(Be * d / (constants.GYRO * A)) # mm
        
        self.set_param_bounds(self.param.slice_thickness, minval=min_thks)

    def update_turbo_factor_bounds(self):
        max_turbo_factor = int(np.floor(self.matrix.value[self.phase_dir.value] * self.partial_Fourier / self.EPI_factor * 2)) # let's limit phase oversampling to 2
        max_turbo_factor = min(max_turbo_factor, 64)

        # turbo_factor must equal 1 when the EPI_factor is even
        if not self.EPI_factor%2:
            self.param.turbo_factor.bounds = (1, 1)
            self.param.turbo_factor.constant = True
        else:
            self.param.turbo_factor.bounds = (1, max_turbo_factor)
            self.param.turbo_factor.constant = False

    def update_EPI_factor_objects(self):
        max_EPI_factor = int(np.floor(self.matrix.value[self.phase_dir.value] * self.partial_Fourier / self.turbo_factor * 2)) # let's limit phase oversampling to 2
        self.set_param_bounds(self.param.EPI_factor, maxval=max_EPI_factor)
        # EPI_factor must be odd for turbo spin echo (GRASE)
        if self.turbo_factor > 1:
            self.param.EPI_factor.objects = [v for v in self.param.EPI_factor.objects if v%2]

    def centermost_echoes_linear_order_func(self, central_segments, reverse_linear_order, num_segm, turbo_factor):
        # get index lists of rf echo(es) and gradient echo(es) closest to k-space center for linear k-space ordering
        centermost_gr_echoes = []
        centermost_rf_echoes = []
        central_indices = central_segments.copy()
        if reverse_linear_order:
            central_indices = [num_segm - 1 - segm for segm in central_indices]
        for segm in central_indices:
            centermost_gr_echoes.append(segm // turbo_factor)
            centermost_rf_echoes.append(segm % turbo_factor)
        return centermost_gr_echoes, centermost_rf_echoes

    def readtrain_spacing_linear_order_func(self, centermost_echoes_linear_order, gr_echo_spacing, EPI_factor, TE):
        centermost_gr_echoes, centermost_rf_echoes = centermost_echoes_linear_order
        readtrain_shift = gr_echo_spacing * (np.mean(centermost_gr_echoes) - (EPI_factor-1)/2)
        central_rf_echo_time = TE - readtrain_shift
        readtrain_spacing = central_rf_echo_time / (1 + np.mean(centermost_rf_echoes))
        return readtrain_spacing

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
        
    def centermost_rf_echo_func(self, EPI_factor, is_gradient_echo, TE, min_readtrain_spacing, split_center, turbo_factor):
        if EPI_factor > 1 or is_gradient_echo:
            return None
        centermost_rf_echo = int(np.floor(TE / min_readtrain_spacing - (1 + .5 * split_center)))
        return min(centermost_rf_echo, turbo_factor - 1 - split_center)
            
    def readtrain_spacing_func(self, EPI_factor, readtrain_spacing_linear_order, TE, centermost_rf_echo, split_center):
        # Equals center position of gradient echo (train) for gradient echo sequences
        # Equals rf echo spacing for spin echo sequences
        if EPI_factor > 1: # linear k-space order for EPI / GRASE
            return readtrain_spacing_linear_order.copy()
        # (turbo) spin echo
        self.readtrain_spacing = TE / (centermost_rf_echo + (1 + .5 * split_center))
        
    def k_read_axis_func(self, freq_dir, FOV, matrix, is_radial, phantom, radial_FOV_oversampling):
        voxel_size = FOV[freq_dir] / matrix[freq_dir]
        if not is_radial:
            num_samples = matrix[freq_dir]
            # at least Nyquist sampling wrt phantom if loaded
            if FOV[freq_dir] < phantom['support'][freq_dir]:
                num_samples = int(np.ceil(phantom['support'][freq_dir] / voxel_size))
        else:
            maxFOV = max(max(phantom['support']), max(FOV))
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

    def set_homodyne_visibility(self, num_blank_lines, is_radial):
        self.param.homodyne.precedence = -1 if (num_blank_lines == 0 or is_radial) else 1

    def lines_to_measure_func(self, k_phase_axis, num_measured_lines):
        lines_to_measure = np.ones(len(k_phase_axis), dtype=bool)
        # undersample by partial Fourier:
        lines_to_measure[num_measured_lines:] = False
        assert(sum(lines_to_measure) == num_measured_lines)
        return lines_to_measure

    def num_sym_segm_func(self, split_center, num_sym_lines, num_blades, num_shots):
        # number of k-space segments symmetric about center:
        if split_center:
            return int(num_sym_lines * num_blades / num_shots)
        return int(np.round((num_sym_lines * num_blades / num_shots - 1) / 2)) * 2 + 1

    def central_segments_func(self, split_center, num_segm, num_sym_segm):
        if split_center:
            return [num_segm - num_sym_segm//2 - 1, num_segm - num_sym_segm//2]
        return [num_segm - num_sym_segm//2 - 1]
    
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

    def set_shot_bounds(self, num_shots):
        self.param.shot.bounds = (1, num_shots)
        self.shot = min(self.shot, num_shots)

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
    
    def k_grid_axes_func(self, is_radial, k_axes, FOV, matrix):
        if not is_radial:
            return k_axes.copy()
        k_grid_axes = [None, None]
        for dim in range(2):
            voxel_size = FOV[dim] / matrix[dim]
            matrix_dim = int(np.ceil(max(FOV[dim], phantom['support'][dim]) / voxel_size))
            k_grid_axes[dim] = recon.get_k_axis(matrix_dim, voxel_size)
        return k_grid_axes
    
    def plain_kspace_comps_func(self, is_radial, phantom, k_grid_axes, k_samples):
        if not is_radial:
            return recon.resample_kspace_Cartesian(phantom, k_grid_axes, shape=k_samples.shape[:-1])
        return recon.resample_kspace(phantom, k_samples)
        
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
        sampling_time = np.expand_dims(sampling_time, axis=[dim for dim in range(3) if dim != freq_dir])
        time_after_excitation = sampling_time + np.expand_dims(TEs, axis=[dim for dim in range(3) if dim != phase_dir])
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

    def PD_and_T1w_func(self, sequence, TR, TE, TI, FA, field_strength, tissues):
        return {component: get_PD_and_T1w(component, sequence, TR, TE, TI, FA, field_strength) for component in set(tissues).union(set(constants.FAT_RESONANCES.keys()))}

    def set_reference_SNR(self, event=None):
        self.reference_SNR = self.SNR

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
        if homodyne:
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

    def RF_inversion_floating_func(self, sequence):
        if not sequence == 'Inversion Recovery':
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

    def slice_select_inversion_floating_func(self, sequence, RF_inversion_floating, slice_thickness):
        if sequence=='Inversion Recovery':
            return None
        flat_dur = RF_inversion_floating['dur_f']
        amp = RF_inversion_floating['FWHM_f'] / (self.inversion_thk_factor * slice_thickness * constants.GYRO)
        return sequence.get_gradient('slice', max_amp=amp, flat_dur=flat_dur, name='slice select inversion', max_slew=self.max_slew)
    
    def inversion_spoiler_floating_func(self, sequence):
        if sequence=='Inversion Recovery':
            return None
        spoiler_area = 30. # uTs/m
        return sequence.get_gradient('slice', total_area=spoiler_area, name='inversion spoiler', max_amp=self.max_amp, max_slew=self.max_slew)

    def readouts_floating_func(self, k_read_axis, pixel_bandwidth, matrix_F, FOV_F, turbo_factor, EPI_factor):
        pixel_size = (len(k_read_axis.value)-1) / len(k_read_axis.value) / (max(k_read_axis.value)-min(k_read_axis.value))
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
        if (EPI_factor==0):
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
            slice_select_refocusing.append(grad.copy())
            time = get_readtrain_pos(readtrain_spacing, rf_echo) - readtrain_spacing/2
            sequence.move_waveform(slice_select_refocusing[rf_echo], time)
        return slice_select_refocusing

    def RF_refocusing_func(self, RF_refocusing_floating, readtrain_spacing):
        if RF_refocusing_floating is None:
            return None
        RF_refocusing = []
        for rf_echo, RF in enumerate(RF_refocusing_floating):
            RF_refocusing.append(RF.copy())
            time = get_readtrain_pos(readtrain_spacing, rf_echo) - readtrain_spacing/2
            sequence.move_waveform(RF_refocusing[rf_echo], time)
        return RF_refocusing

    def slice_select_inversion_func(self, slice_select_inversion_floating, TI):
        if slice_select_inversion_floating is None:
            return None
        slice_select_inversion = slice_select_inversion_floating.copy()
        sequence.move_waveform(slice_select_inversion, -TI)
        return slice_select_inversion
        
    def RF_inversion_func(self, RF_inversion_floating, TI):
        if RF_inversion_floating is None:
            return None
        RF_inversion = RF_inversion_floating.copy()
        sequence.move_waveform(RF_inversion, -TI)
        return RF_inversion
    
    def inversion_spoiler_func(self, inversion_spoiler_floating, RF_inversion):
        if inversion_spoiler_floating is None:
            return None
        inversion_spoiler = inversion_spoiler_floating.copy()
        time = RF_inversion['time'][-1] + inversion_spoiler['dur_f']/2
        sequence.move_waveform(inversion_spoiler, time)
        return inversion_spoiler

    def FatSat_spoiler_func(self, FatSat_spoiler_floating, slice_select_excitation):
        if FatSat_spoiler_floating is None:
            return None
        FatSat_spoiler = FatSat_spoiler_floating.copy()
        time = slice_select_excitation['time'][0] - FatSat_spoiler['dur_f']/2
        sequence.move_waveform(FatSat_spoiler, time)
        return FatSat_spoiler

    def RF_FatSat_func(self, RF_FatSat_floating, FatSat_spoiler_floating):
        if RF_FatSat_floating is None:
            return None
        RF_FatSat = RF_FatSat_floating.copy()
        t = FatSat_spoiler_floating['time'][0] - RF_FatSat['dur_f']/2
        sequence.move_waveform(RF_FatSat, t)
        return RF_FatSat

    def readouts_func(self, turbo_factor, readtrain_spacing, EPI_factor, gr_echo_spacing, readouts_floating):
        readouts = []
        for rf_echo in range(turbo_factor):
            readouts.append([])
            readtrain_pos = get_readtrain_pos(readtrain_spacing, rf_echo)
            for gr_echo in range(EPI_factor):
                readout = readouts_floating[rf_echo][gr_echo].copy()
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
                sampling_window = sampling_windows_floating[rf_echo][gr_echo].copy()
                pos = readtrain_pos + (gr_echo - (EPI_factor-1) / 2) * gr_echo_spacing
                sequence.move_waveform(sampling_window, pos)
                sampling_windows[-1].append(sampling_window)
        return sampling_windows
    
    def read_prephaser_func(self, read_prephaser_floating, is_gradient_echo, readouts, RF_excitation):
        read_prephaser = read_prephaser_floating.copy()
                
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
            phaser = phasers_floating[rf_echo].copy()
            readtrain_pos = get_readtrain_pos(readtrain_spacing, rf_echo)
            phaser_time = readtrain_pos - (gre_echo_train_dur + phaser['dur_f'])/2 + readout_risetime
            sequence.move_waveform(phaser, phaser_time)
            phasers.append(phaser)
        return phasers
    
    def rephasers_func(self, turbo_factor, readtrain_spacing, gre_echo_train_dur, readout_risetime, rephasers_floating):
        rephasers = []
        for rf_echo in range(turbo_factor):
            readtrain_pos = get_readtrain_pos(readtrain_spacing, rf_echo)
            rephaser = rephasers_floating[rf_echo].copy()
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
                blip = blips_floating[rf_echo][gr_echo].copy()
                blip_time = readtrain_pos + gr_echo_spacing * (gr_echo - EPI_factor/2 + 1)
                sequence.move_waveform(blip, blip_time)
                blips[-1].append(blip)
        return blips

    def spoiler_func(self, readouts, spoiler_floating):
        spoiler = spoiler_floating.copy()
        spoiler_time = readouts[-1][-1]['center_f'] + (readouts[-1][-1]['flat_dur_f'] + spoiler['dur_f']) / 2
        sequence.move_waveform(spoiler, spoiler_time)
        return spoiler

    # End of node funcs

    def add_signals(self):
        self.boards['signal']['objects']['signals'] = []
        scale = 1/np.max(np.abs(np.real(self.measured_kspace)))
        signal_exponent = .5
        spoke = self.shot-1 if self.is_radial.value else 0
        for rf_echo in range(self.turbo_factor):
            signals = []
            for gr_echo in range(self.EPI_factor):
                ky = self.pe_table[self.shot-1, rf_echo, gr_echo]
                waveform = np.real(np.take(self.measured_kspace[..., spoke], indices=ky, axis=self.phase_dir.value))
                t = np.take(self.time_after_excitation[..., spoke if spoke<self.time_after_excitation.shape[-1] else 0], indices=ky, axis=self.phase_dir.value)
                signal = sequence.get_signal(waveform, t, scale, signal_exponent)
                signals.append(signal)
            self.boards['signal']['objects']['signals'].append(signals)

    def update_k_line_coords(self, attr, old, hover_index):
        if len(hover_index['index']) > 0:
            object = self.boards[hover_index['board'][0]]['object_list'][hover_index['index'][0]]
            self.k_line.event(coords=list(self.get_k_on_interval(object['time'][[0, -1]])))
        else:
            self.k_line.event(coords=[None])

    def get_hover_tool(self, board, obj):
        attributes = [attr for attr in obj.keys() if attr not in ['time', board] and '_f' not in attr]
        if board in ['frequency', 'phase', 'RF', 'ADC']:
            with open(Path(__file__).parent / 'hoverCallback.js', 'r') as file:
                hover_callback = CustomJS(args={'hover_index': self.hover_index, 'board': board}, code=file.read())
        else:
            hover_callback = None
        hover = HoverTool(tooltips=[(attr, f'@{attr}') for attr in attributes], attachment='below', callback=hover_callback)
        return hover, attributes

    def render_polygons(self, board, hoverTool=True):
        if self.boards[board]['objects']:
            object_list = flatten_dicts(self.boards[board]['objects'].values())
            self.boards[board]['object_list'] = object_list
            self.board_plots[board]['area'] = hv.Area(sequence.accumulate_waveforms(object_list, board), self.time_dim, self.boards[board]['dim']).opts(color=self.boards[board]['color'])
            hover, attributes = self.get_hover_tool(board, object_list[0])
            tools = [hover] if hoverTool else []
            self.board_plots[board]['polygons'] = hv.Polygons(object_list, kdims=[self.time_dim, self.boards[board]['dim']], vdims=attributes).opts(tools=tools, cmap=[self.boards[board]['color']], hooks=[hideframe_hook, partial(bounds_hook, xbounds=(-19000, 19000))])

    def render_ADC_windows(self, hoverTool=True):
        object_list = flatten_dicts(self.boards['ADC']['objects'].values())
        self.boards['ADC']['object_list'] = object_list
        hover, attributes = self.get_hover_tool('ADC', object_list[0])
        tools = [hover] if hoverTool else []
        for obj in object_list:
            obj.update({'c1': obj['time'][0], 'c2': -2, 'c3': obj['time'][-1], 'c4': 2})
        self.board_plots['signal']['ADC'] = hv.Rectangles(object_list, kdims=['c1', 'c2', 'c3', 'c4'], vdims=attributes).opts(tools=tools)

    def render_TR_span(self):
        t0 = self.get_sequence_start()
        for board in ['frequency', 'phase', 'slice', 'RF', 'signal']:
            self.board_plots[board]['TRspan'] = hv.VSpan(-20000, t0, kdims=[self.time_dim, self.boards[board]['dim']]).opts(color='gray', fill_alpha=.3)
            self.board_plots[board]['TRspan'] *= hv.VSpan(t0 + self.TR, 20000, kdims=[self.time_dim, self.boards[board]['dim']]).opts(color='gray', fill_alpha=.3)

    def render_frequency_board(self):
        self.render_polygons('frequency', hoverTool=True)
        add_to_pipeline(self.sequence_plot_pipeline, ['calculate_k_trajectory'])

    def render_phase_board(self):
        self.render_polygons('phase', hoverTool=True)
        add_to_pipeline(self.sequence_plot_pipeline, ['calculate_k_trajectory'])

    def render_slice_board(self):
        self.render_polygons('slice', hoverTool=True)

    def render_RF_board(self):
        self.render_polygons('RF', hoverTool=True)
        add_to_pipeline(self.sequence_plot_pipeline, ['calculate_k_trajectory'])

    def render_signal_board(self):
        self.add_signals()
        self.render_polygons('signal', hoverTool=False)
        self.render_ADC_windows(hoverTool=True)
        add_to_pipeline(self.sequence_plot_pipeline, ['calculate_k_trajectory'])

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
        t = np.concatenate((*(self.board_plots[board]['area']['time'] for board in ['frequency', 'phase']), [t for ref in refocus_intervals for t in ref])) # k event times
        t = np.unique(np.concatenate((t, np.arange(0., max(t), dt)))) # merge with time grid
        kx = self.get_k_coords(t, *(self.board_plots['frequency']['area'][dim] for dim in ['G read', 'time']), refocus_intervals)
        ky = self.get_k_coords(t, *(self.board_plots['phase']['area'][dim] for dim in ['G phase', 'time']), refocus_intervals)
        if not self.is_radial.value:
            if self.phase_dir.value==1:
                kx, ky = ky, kx
        else: # rotate by spoke/blade angle
            angle = np.radians(self.spoke_angle_node.value)
            cos, sin = np.cos(angle), np.sin(angle)
            kx, ky = cos * kx - sin * ky, sin * kx + cos * ky
        self.k_trajectory = {'kx': kx, 'ky': ky, 't': t, 'dt': dt}

    @param.depends('sequence', 'FatSat', 'TR', 'TE', 'FA', 'TI', 'FOV_F', 'FOV_P', 'phase_oversampling', 'num_shots', 'matrix_F', 'matrix_P', 'slice_thickness', 'trajectory', 'frequency_direction', 'pixel_bandwidth', 'partial_Fourier', 'turbo_factor', 'EPI_factor', 'shot')
    def display_sequence_plot(self):
        self.run_sequence_plot_pipeline()
        last = len(self.board_plots)-1
        self.sequence_plot = hv.Layout(list([hv.Overlay(list(board_plot.values())).opts(width=1700, height=180 if n==last else 120, border=0, xaxis='bottom' if n==last else None) for n, board_plot in enumerate(self.board_plots.values())])).cols(1).options(toolbar='below')
        return self.sequence_plot

    @param.depends('object', 'field_strength', 'sequence', 'FatSat', 'TR', 'TE', 'FA', 'TI', 'FOV_F', 'FOV_P', 'phase_oversampling', 'num_shots', 'matrix_F', 'matrix_P', 'recon_matrix_F', 'recon_matrix_P', 'slice_thickness', 'trajectory', 'frequency_direction', 'pixel_bandwidth', 'NSA', 'partial_Fourier', 'turbo_factor', 'EPI_factor', 'kspace_type', 'show_processed_kspace', 'kspace_exponent', 'homodyne', 'do_apodize', 'apodization_alpha', 'do_zerofill', 'radial_FOV_oversampling')
    def display_kspace(self):
        operator = constants.OPERATORS[self.kspace_type]
        if self.show_processed_kspace:
            self.run_recon_pipeline()
            k_axes = []
            for dim in range(2):
                k_axes.append(recon.get_k_axis(self.oversampled_recon_matrix[dim], self.FOV.value[dim] / self.recon_matrix.value[dim]))
                # half-sample shift axis when odd number of zeroes:
                if (self.oversampled_recon_matrix[dim] - self.full_kspace.shape[dim])%2:
                    shift = self.recon_matrix.value[dim] / (2 * self.oversampled_recon_matrix[dim] * self.FOV.value[dim])
                    k_axes[-1] -= shift
            ksp = xr.DataArray(
                operator(self.zerofilled_kspace**self.kspace_exponent), 
                dims=('ky', 'kx'),
                coords={'kx': k_axes[1], 'ky': k_axes[0]}
            )
        else:
            self.run_acquisition_pipeline()
            ksp = xr.DataArray(
                operator(self.gridded_kspace**self.kspace_exponent), 
                dims=('ky', 'kx'),
                coords={'kx': self.k_grid_axes[1], 'ky': self.k_grid_axes[0]}
            )
        ksp.kx.attrs['units'] = ksp.ky.attrs['units'] = '1/mm'
        lim = 1.12 * max(self.k_grid_axes[1])
        self.kspace_image = hv.Image(ksp, vdims=['magnitude']).opts(xlim=(-lim,lim), ylim=(-lim,lim))
        return self.kspace_image

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

    @param.depends('object', 'field_strength', 'sequence', 'FatSat', 'TR', 'TE', 'FA', 'TI', 'FOV_F', 'FOV_P', 'phase_oversampling', 'num_shots', 'matrix_F', 'matrix_P', 'recon_matrix_F', 'recon_matrix_P', 'slice_thickness', 'trajectory', 'frequency_direction', 'pixel_bandwidth', 'NSA', 'partial_Fourier', 'turbo_factor', 'EPI_factor', 'image_type', 'show_FOV', 'homodyne', 'do_apodize', 'apodization_alpha', 'do_zerofill', 'radial_FOV_oversampling')
    def display_image(self):
        self.run_recon_pipeline()
        return self.image.value * self.FOV_box.value