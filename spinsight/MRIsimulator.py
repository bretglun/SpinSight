import holoviews as hv
from holoviews import streams
import param
import numpy as np
import math
from pathlib import Path
from spinsight import constants, formatting, phantom, nodes
from spinsight.DAG import Graph
from bokeh.models import HoverTool, CustomJS, ColumnDataSource
import warnings
from functools import partial

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


def get_k_on_interval(interval, k_trajectory):
    t = np.arange(*interval[[0, -1]], k_trajectory['dt'])
    kx = np.interp(t, k_trajectory['t'], k_trajectory['kx'])
    ky = np.interp(t, k_trajectory['t'], k_trajectory['ky'])
    return zip(kx, ky)


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
        
        hv.opts.defaults(hv.opts.Image(width=500, height=500, invert_yaxis=False, toolbar='below', cmap='gray', aspect='equal'))
        hv.opts.defaults(hv.opts.HLine(line_width=1.5, line_color='gray'))
        hv.opts.defaults(hv.opts.VSpan(color='orange', fill_alpha=.1, hover_fill_alpha=.8, default_tools=[]))
        hv.opts.defaults(hv.opts.Rectangles(color=constants.BOARD_COLORS['ADC'], line_color=constants.BOARD_COLORS['ADC'], fill_alpha=.1, line_alpha=.3, hover_fill_alpha=.8, default_tools=[]))
        hv.opts.defaults(hv.opts.Box(line_width=3))
        hv.opts.defaults(hv.opts.Ellipse(line_width=3))
        hv.opts.defaults(hv.opts.Area(fill_alpha=.5, line_width=1.5, line_color='gray', default_tools=[]))
        hv.opts.defaults(hv.opts.Polygons(line_width=1.5, fill_alpha=0, line_alpha=0, line_color='gray', selection_line_color='black', hover_fill_alpha=.8, hover_line_alpha=1, selection_fill_alpha=.8, selection_line_alpha=1, nonselection_line_alpha=0, default_tools=[]))
        hv.opts.defaults(hv.opts.Curve(line_width=5, line_color=constants.BOARD_COLORS['ADC']))
        hv.opts.defaults(hv.opts.Points(line_color=None, color=constants.BOARD_COLORS['ADC'], size=15))


        node_specs = {par: {'params': self} for par in self.param if par != 'name'}

        node_specs['set_isotropic_voxel_size'] = {
            'action': True,
            'func': self.set_isotropic_voxel_size,
            'parents': ['is_radial', 'FOV_F', 'matrix_F', 'FOV_P', 'matrix_P']
        }
        
        node_specs['set_TR_bounds'] = {
            'action': True,
            'func': self.set_TR_bounds,
            'parents': ['min_TR']
        }
        
        node_specs['set_TE_bounds'] = {
            'action': True,
            'func': self.set_TE_bounds,
            'parents': ['min_TE', 'max_TE']
        }
        
        node_specs['set_TI_bounds'] = {
            'action': True,
            'func': self.set_TI_bounds,
            'parents': ['sequence_type', 'TR', 'spoiler', 'slice_select_inversion_floating']
        }
        
        node_specs['set_x_y_labels'] = {
            'action': True,
            'func': self.set_x_y_labels,
            'parents': ['frequency_direction']
        }
        
        node_specs['set_labels_by_trajectory'] = {
            'action': True,
            'func': self.set_labels_by_trajectory,
            'parents': ['shot_label']
        }

        node_specs['set_shot_label'] = {
            'action': True,
            'func': self.set_shot_label,
            'parents': ['shot_label']
        }
        
        node_specs['set_spoke_angle'] = {
            'action': True,
            'func': self.set_spoke_angle,
            'parents': ['spoke_angle']
        }
        
        node_specs['set_num_shots'] = {
            'action': True,
            'func': self.set_num_shots,
            'parents': ['num_shots']
        }
        
        node_specs['set_relative_SNR'] = {
            'action': True,
            'func': self.set_relative_SNR,
            'parents': ['relative_SNR']
        }
        
        node_specs['set_scantime'] = {
            'action': True,
            'func': self.set_scantime,
            'parents': ['scantime']
        }
        
        node_specs['set_trajectory_objects'] = {
            'action': True,
            'func': self.set_trajectory_objects,
            'parents': ['EPI_factor', 'turbo_factor']
        }

        node_specs['set_pixel_bandwidth_bounds'] = {
            'action': True,
            'func': self.set_pixel_bandwidth_bounds,
            'parents': ['matrix_F', 'FOV_F', 'is_gradient_echo', 'RF_refocusing', 'turbo_factor', 'EPI_factor', 'TE', 'RF_excitation', 'refocusing_time', 'readtrain_spacing', 'TR', 'sequence_start', 'spoiler', 'k0_echo_indices_linear_order', 'k0_echo_indices_reverse_linear_order', 'reverse_linear_order', 'read_prephaser', 'phaser_duration', 'slice_select_excitation', 'slice_select_rephaser', 'max_blip_dur', 'slice_select_refocusing']
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
        
        node_specs['set_recon_matrix_F_bounds'] = {
            'action': True,
            'func': self.set_recon_matrix_F_bounds,
            'parents': ['matrix_F']
        }
        
        node_specs['set_recon_matrix_P_bounds'] = {
            'action': True,
            'func': self.set_recon_matrix_P_bounds,
            'parents': ['matrix_P']
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
        
        node_specs['set_parameter_style_visibility'] = {
            'action': True,
            'func': self.set_parameter_style_visibility,
            'parents': ['parameter_style']
        }
        
        node_specs['set_partial_Fourier_visibility'] = {
            'action': True,
            'func': self.set_partial_Fourier_visibility,
            'parents': ['is_radial']
        }
        
        node_specs['set_frequency_direction_visibility'] = {
            'action': True,
            'func': self.set_frequency_direction_visibility,
            'parents': ['is_radial']
        }
        
        node_specs['set_phase_oversampling_visibility'] = {
            'action': True,
            'func': self.set_phase_oversampling_visibility,
            'parents': ['is_radial']
        }
        
        node_specs['set_radial_factor_visibility'] = {
            'action': True,
            'func': self.set_radial_factor_visibility,
            'parents': ['is_radial']
        }
        
        node_specs['set_TI_visibility'] = {
            'action': True,
            'func': self.set_TI_visibility,
            'parents': ['sequence_type']
        }
        
        node_specs['set_FA_visibility'] = {
            'action': True,
            'func': self.set_FA_visibility,
            'parents': ['sequence_type']
        }
        
        node_specs['set_turbo_factor_visibility'] = {
            'action': True,
            'func': self.set_turbo_factor_visibility,
            'parents': ['sequence_type']
        }
        
        node_specs['set_homodyne_visibility'] = {
            'action': True,
            'func': self.set_homodyne_visibility,
            'parents': ['num_blank_lines', 'is_radial']
        }
        
        node_specs['set_apodization_alpha_visibility'] = {
            'action': True,
            'func': self.set_apodization_alpha_visibility,
            'parents': ['do_apodize']
        }

        node_specs['phantom_object'] = {
            'func': nodes.phantom_object_func,
            'parents': ['object', 'min_voxel_size']
        }
        
        node_specs['tissues'] = {
            'func': nodes.tissues_func,
            'parents': ['phantom_object']
        }

        node_specs['is_radial'] = {
            'func': nodes.is_radial_func,
            'parents': ['trajectory']
        }
        
        node_specs['is_gradient_echo'] = {
            'func': nodes.is_gradient_echo_func,
            'parents': ['sequence_type']
        }

        node_specs['freq_dir'] = {
            'func': nodes.freq_dir_func,
            'parents': ['frequency_direction', 'is_radial']
        }

        node_specs['phase_dir'] = {
            'func': nodes.phase_dir_func,
            'parents': ['freq_dir']
        }
        
        node_specs['FOV'] = {
            'func': nodes.FOV_func,
            'parents': ['FOV_F', 'FOV_P', 'freq_dir']
        }

        node_specs['matrix'] = {
            'func': nodes.matrix_func,
            'parents': ['matrix_F', 'matrix_P', 'freq_dir']
        }

        node_specs['recon_matrix'] = {
            'func': nodes.recon_matrix_func,
            'parents': ['recon_matrix_P', 'recon_matrix_F', 'freq_dir', 'do_zerofill', 'matrix']
        }

        node_specs['rec_acq_ratio_F'] = {
            'func': nodes.rec_acq_ratio_F_func,
            'parents': ['recon_matrix_F', 'matrix_F']
        }

        node_specs['rec_acq_ratio_P'] = {
            'func': nodes.rec_acq_ratio_P_func,
            'parents': ['recon_matrix_P', 'matrix_P']
        }

        node_specs['RF_excitation'] = {
            'func': nodes.RF_excitation_func,
            'parents': ['FA', 'is_gradient_echo']
        }

        node_specs['RF_refocusing_floating'] = {
            'func': nodes.RF_refocusing_floating_func,
            'parents': ['is_gradient_echo', 'turbo_factor']
        }

        node_specs['RF_inversion_floating'] = {
            'func': nodes.RF_inversion_floating_func,
            'parents': ['sequence_type']
        }

        node_specs['RF_FatSat_floating'] = {
            'func': nodes.RF_FatSat_floating_func,
            'parents': ['FatSat', 'field_strength']
        }

        node_specs['FatSat_spoiler_floating'] = {
            'func': nodes.FatSat_spoiler_floating_func,
            'parents': ['FatSat']
        }

        node_specs['slice_select_excitation'] = {
            'func': nodes.slice_select_excitation_func,
            'parents': ['RF_excitation', 'slice_thickness']
        }

        node_specs['slice_select_rephaser'] = {
            'func': nodes.slice_select_rephaser_func,
            'parents': ['slice_select_excitation']
        }

        node_specs['slice_select_refocusing_floating'] = {
            'func': nodes.slice_select_refocusing_floating_func,
            'parents': ['RF_refocusing_floating', 'slice_thickness', 'turbo_factor']
        }

        node_specs['slice_select_inversion_floating'] = {
            'func': nodes.slice_select_inversion_floating_func,
            'parents': ['sequence_type', 'RF_inversion_floating', 'slice_thickness']
        }

        node_specs['inversion_spoiler_floating'] = {
            'func': nodes.inversion_spoiler_floating_func,
            'parents': ['sequence_type']
        }

        node_specs['readouts_floating'] = {
            'func': nodes.readouts_floating_func,
            'parents': ['k_read_axis', 'pixel_bandwidth', 'matrix_F', 'FOV_F', 'turbo_factor', 'EPI_factor']
        }

        node_specs['sampling_windows_floating'] = {
            'func': nodes.sampling_windows_floating_func,
            'parents': ['turbo_factor', 'EPI_factor', 'readouts_floating']
        }

        node_specs['readout_risetime'] = {
            'func': nodes.readout_risetime_func,
            'parents': ['readouts_floating']
        }

        node_specs['read_prephaser_floating'] = {
            'func': nodes.read_prephaser_floating_func,
            'parents': ['readouts_floating', 'is_gradient_echo']
        }

        node_specs['phase_step_area'] = {
            'func': nodes.phase_step_area_func,
            'parents': ['k_phase_axis']
        }
        
        node_specs['largest_phaser_area'] = {
            'func': nodes.largest_phaser_area_func,
            'parents': ['k_phase_axis']
        }
        
        node_specs['phaser_duration'] = {
            'func': nodes.phaser_duration_func,
            'parents': ['largest_phaser_area']
        }
        
        node_specs['max_blip_dur'] = {
            'func': nodes.max_blip_dur_func,
            'parents': ['EPI_factor', 'phase_step_area', 'num_shots', 'turbo_factor']
        }
        
        node_specs['readout_gap'] = {
            'func': nodes.readout_gap_func,
            'parents': ['max_blip_dur', 'readouts_floating']
        }
        
        node_specs['gr_echo_spacing'] = {
            'func': nodes.gr_echo_spacing_func,
            'parents': ['readouts_floating', 'readout_gap']
        }
        
        node_specs['gre_echo_train_dur'] = {
            'func': nodes.gre_echo_train_dur_func,
            'parents': ['EPI_factor', 'gr_echo_spacing', 'readout_gap']
        }
        
        node_specs['phasers_floating'] = {
            'func': nodes.phasers_floating_func,
            'parents': ['turbo_factor', 'largest_phaser_area', 'pe_table', 'phase_step_area', 'shot']
        }
        
        node_specs['blips_floating'] = {
            'func': nodes.blips_floating_func,
            'parents': ['turbo_factor', 'EPI_factor', 'phase_step_area', 'pe_table', 'shot']
        }
        
        node_specs['rephasers_floating'] = {
            'func': nodes.rephasers_floating_func,
            'parents': ['turbo_factor', 'phasers_floating', 'blips_floating', 'largest_phaser_area']
        }
        
        node_specs['spoiler_floating'] = {
            'func': nodes.spoiler_floating_func,
            'parents': []
        }
        
        node_specs['readtrain_center_time'] = {
            'func': nodes.readtrain_center_time_func,
            'parents': ['readtrain_spacing', 'turbo_factor']
        }

        node_specs['readout_center_time'] = {
            'func': nodes.readout_center_time_func,
            'parents': ['EPI_factor', 'gr_echo_spacing', 'readtrain_center_time']
        }

        node_specs['refocusing_time'] = {
            'func': nodes.refocusing_time_func,
            'parents': ['TE', 'readtrain_center_time', 'readtrain_spacing']
        }
        
        node_specs['slice_select_refocusing'] = {
            'func': nodes.slice_select_refocusing_func,
            'parents': ['slice_select_refocusing_floating', 'refocusing_time']
        }
        
        node_specs['RF_refocusing'] = {
            'func': nodes.RF_refocusing_func,
            'parents': ['RF_refocusing_floating', 'refocusing_time']
        }
        
        node_specs['slice_select_inversion'] = {
            'func': nodes.slice_select_inversion_func,
            'parents': ['slice_select_inversion_floating', 'TI']
        }
        
        node_specs['RF_inversion'] = {
            'func': nodes.RF_inversion_func,
            'parents': ['RF_inversion_floating', 'TI']
        }
        
        node_specs['inversion_spoiler'] = {
            'func': nodes.inversion_spoiler_func,
            'parents': ['inversion_spoiler_floating', 'RF_inversion']
        }
        
        node_specs['FatSat_spoiler'] = {
            'func': nodes.FatSat_spoiler_func,
            'parents': ['FatSat_spoiler_floating', 'slice_select_excitation']
        }
        
        node_specs['RF_FatSat'] = {
            'func': nodes.RF_FatSat_func,
            'parents': ['RF_FatSat_floating', 'FatSat_spoiler_floating']
        }
        
        node_specs['readouts'] = {
            'func': nodes.readouts_func,
            'parents': ['readouts_floating', 'readout_center_time']
        }

        node_specs['sampling_windows'] = {
            'func': nodes.sampling_windows_func,
            'parents': ['sampling_windows_floating', 'readout_center_time']
        }

        node_specs['read_prephaser'] = {
            'func': nodes.read_prephaser_func,
            'parents': ['read_prephaser_floating', 'is_gradient_echo', 'readouts', 'RF_excitation']
        }

        node_specs['phasers'] = {
            'func': nodes.phasers_func,
            'parents': ['readtrain_center_time', 'phasers_floating', 'gre_echo_train_dur', 'readout_risetime']
        }

        node_specs['rephasers'] = {
            'func': nodes.rephasers_func,
            'parents': ['readtrain_center_time', 'gre_echo_train_dur', 'readout_risetime', 'rephasers_floating']
        }
        
        node_specs['blips'] = {
            'func': nodes.blips_func,
            'parents': ['readtrain_center_time', 'EPI_factor', 'gr_echo_spacing', 'blips_floating']
        }
        
        node_specs['spoiler'] = {
            'func': nodes.spoiler_func,
            'parents': ['readouts', 'spoiler_floating']
        }
        
        node_specs['sequence_start'] = {
            'func': nodes.sequence_start_func,
            'parents': ['slice_select_inversion', 'RF_FatSat', 'slice_select_excitation']
        }

        node_specs['signal_curves'] = {
            'func': nodes.signal_curves_func,
            'parents': ['measured_kspace', 'shot', 'is_radial', 'turbo_factor', 'EPI_factor', 'pe_table', 'phase_dir', 'time_after_excitation']
        }

        node_specs['k_read_axis'] = {
            'func': nodes.k_read_axis_func,
            'parents': ['freq_dir', 'FOV', 'matrix', 'is_radial', 'phantom_object', 'radial_FOV_oversampling']
        }
        
        node_specs['min_TE'] = {
            'func': nodes.min_TE_func,
            'parents': ['k0_echo_indices_linear_order', 'k0_echo_indices_reverse_linear_order', 'min_refocusing_time', 'min_RF_to_readtrain_center', 'gr_echo_spacing', 'EPI_factor', 'is_gradient_echo', 'turbo_factor']
        }
        
        node_specs['max_TE'] = {
            'func': nodes.max_TE_func,
            'parents': ['TR', 'sequence_start', 'spoiler_floating', 'readout_risetime', 'gre_echo_train_dur', 'EPI_factor', 'turbo_factor', 'k0_echo_indices_linear_order', 'k0_echo_indices_reverse_linear_order', 'gr_echo_spacing']
        }
        
        node_specs['min_TR'] = {
            'func': nodes.min_TR_func,
            'parents': ['spoiler', 'sequence_start']
        }
        
        node_specs['max_readout_area'] = {
            'func': nodes.max_readout_area_func,
            'parents': ['pixel_bandwidth', 'is_gradient_echo', 'k0_gr_echo_index', 'TE', 'RF_excitation', 'phaser_duration', 'slice_select_excitation', 'slice_select_rephaser', 'max_blip_dur', 'readtrain_spacing', 'refocusing_time', 'RF_refocusing_floating', 'EPI_factor']
        }

        node_specs['max_phaser_area'] = {
            'func': nodes.max_phaser_area_func,
            'parents': ['is_gradient_echo', 'readtrain_spacing', 'RF_excitation', 'gre_echo_train_dur', 'readout_risetime', 'refocusing_time', 'RF_refocusing']
        }

        node_specs['min_refocusing_time'] = {
            'func': nodes.min_refocusing_time_func,
            'parents': ['is_gradient_echo', 'RF_excitation', 'RF_refocusing_floating', 'read_prephaser_floating', 'slice_select_excitation', 'slice_select_rephaser', 'slice_select_refocusing_floating']
        }

        node_specs['min_RF_to_readtrain_center'] = {
            'func': nodes.min_RF_to_readtrain_center_func,
            'parents': ['is_gradient_echo', 'RF_excitation', 'read_prephaser_floating', 'slice_select_excitation', 'slice_select_rephaser', 'RF_refocusing_floating', 'slice_select_refocusing_floating', 'gre_echo_train_dur', 'readout_risetime', 'phaser_duration']
        }

        node_specs['k0_echo_indices_linear_order'] = {
            'func': nodes.k0_echo_indices_linear_order_func,
            'parents': ['k0_segment', 'turbo_factor']
        }

        node_specs['k0_echo_indices_reverse_linear_order'] = {
            'func': nodes.k0_echo_indices_reverse_linear_order_func,
            'parents': ['k0_echo_indices_linear_order', 'EPI_factor', 'turbo_factor']
        }

        node_specs['k0_index'] = {
            'func': nodes.k0_index_func,
            'parents': ['k0_echo_indices_reverse_linear_order', 'k0_echo_indices_linear_order', 'is_gradient_echo', 'turbo_factor', 'TE', 'EPI_factor', 'gr_echo_spacing', 'min_refocusing_time', 'min_RF_to_readtrain_center']
        }
        
        node_specs['k0_rf_echo_index'] = {
            'func': nodes.k0_rf_echo_index_func,
            'parents': ['k0_index']
        }
        
        node_specs['k0_gr_echo_index'] = {
            'func': nodes.k0_gr_echo_index_func,
            'parents': ['k0_index']
        }
        
        node_specs['reverse_linear_order'] = {
            'func': nodes.reverse_linear_order_func,
            'parents': ['k0_index']
        }
        
        node_specs['readtrain_spacing'] = {
            'func': nodes.readtrain_spacing_func,
            'parents': ['EPI_factor', 'gr_echo_spacing', 'TE', 'k0_gr_echo_index', 'k0_rf_echo_index']
        }

        node_specs['num_blades'] = {
            'func': nodes.num_blades_func,
            'parents': ['is_radial', 'matrix', 'radial_factor', 'turbo_factor', 'EPI_factor']
        }

        node_specs['k_angles'] = {
            'func': nodes.k_angles_func,
            'parents': ['num_blades']
        }

        node_specs['spoke_angle'] = {
            'func': nodes.spoke_angle_func,
            'parents': ['k_angles', 'shot']
        }

        node_specs['num_shots'] = {
            'func': nodes.num_shots_func,
            'parents': ['matrix_P', 'phase_oversampling', 'partial_Fourier', 'turbo_factor', 'EPI_factor', 'is_radial', 'num_blades']
        }

        node_specs['shot_label'] = {
            'func': nodes.shot_label_func,
            'parents': ['is_radial', 'EPI_factor', 'turbo_factor']
        }

        node_specs['num_measured_lines'] = {
            # measured lines per blade
            'func': nodes.num_measured_lines_func,
            'parents': ['turbo_factor', 'EPI_factor', 'num_shots', 'is_radial']
        }

        node_specs['k_phase_axis'] = {
            'func': nodes.k_phase_axis_func,
            'parents': ['is_radial', 'num_measured_lines', 'matrix', 'phase_dir', 'phase_oversampling', 'FOV']
        }

        node_specs['num_blank_lines'] = {
            'func': nodes.num_blank_lines_func,
            'parents': ['k_phase_axis', 'lines_to_measure']
        }

        node_specs['lines_to_measure'] = {
            'func': nodes.lines_to_measure_func,
            'parents': ['k_phase_axis', 'num_measured_lines']
        }

        node_specs['num_segm'] = {
            'func': nodes.num_segm_func,
            'parents': ['turbo_factor', 'EPI_factor']
        }

        node_specs['num_sym_lines'] = {
            'func': nodes.num_sym_lines_func,
            'parents': ['num_measured_lines', 'k_phase_axis']
        }

        node_specs['num_sym_segm'] = {
            'func': nodes.num_sym_segm_func,
            'parents': ['num_segm', 'num_sym_lines', 'num_measured_lines']
        }

        node_specs['k0_segment'] = {
            'func': nodes.k0_segment_func,
            'parents': ['num_segm', 'num_sym_segm']
        }

        node_specs['pe_table'] = {
            'func': nodes.pe_table_func,
            'parents': ['num_measured_lines', 'num_segm', 'num_shots', 'EPI_factor', 'turbo_factor', 'num_sym_segm', 'k0_rf_echo_index', 'reverse_linear_order']
        }

        node_specs['signal_level'] = {
            'func': nodes.signal_level_func,
            'parents': ['k_read_axis', 'lines_to_measure', 'num_blades', 'slice_thickness', 'FOV', 'matrix']
        }

        node_specs['spin_echoes'] = {
            'func': nodes.spin_echoes_func,
            'parents': ['lines_to_measure', 'pe_table', 'readtrain_spacing']
        }

        node_specs['sampling_time'] = {
            'func': nodes.sampling_time_func,
            'parents': ['pixel_bandwidth', 'k_read_axis']
        }

        node_specs['time_after_excitation'] = {
            'func': nodes.time_after_excitation_func,
            'parents': ['lines_to_measure', 'pe_table', 'readouts', 'sampling_time', 'freq_dir', 'phase_dir']
        }

        node_specs['time_relative_inphase'] = {
            'func': nodes.time_relative_inphase_func,
            'parents': ['time_after_excitation', 'is_gradient_echo', 'spin_echoes', 'phase_dir']
        }

        node_specs['dephasing'] = {
            'func': nodes.dephasing_func,
            'parents': ['field_strength', 'time_relative_inphase']
        }

        node_specs['T2w'] = {
            'func': nodes.T2w_func,
            'parents': ['tissues', 'time_after_excitation', 'time_relative_inphase', 'field_strength']
        }

        node_specs['k_axes'] = {
            'func': nodes.k_axes_func,
            'parents': ['freq_dir', 'phase_dir', 'k_read_axis', 'k_phase_axis', 'lines_to_measure']
        }
        
        node_specs['k_grid_axes'] = {
            'func': nodes.k_grid_axes_func,
            'parents': ['is_radial', 'k_axes', 'FOV', 'matrix', 'phantom_object']
        }
        
        node_specs['k_samples'] = {
            'func': nodes.k_samples_func,
            'parents': ['k_axes', 'k_angles']
        }
        
        node_specs['plain_kspace_comps'] = {
            'func': nodes.plain_kspace_comps_func,
            'parents': ['is_radial', 'phantom_object', 'k_grid_axes', 'k_samples']
        }
        
        node_specs['thick_kspace_comps'] = {
            'func': nodes.thick_kspace_comps_func,
            'parents': ['slice_thickness', 'k_samples', 'plain_kspace_comps']
        }
        
        node_specs['kspace_comps'] = {
            'func': nodes.kspace_comps_func,
            'parents': ['tissues', 'thick_kspace_comps', 'T2w', 'dephasing']
        }
        
        node_specs['decayed_signal'] = {
            'func': nodes.decayed_signal_func,
            'parents': ['signal_level', 'T2w', 'reference_tissue', 'k_read_axis', 'k_phase_axis', 'freq_dir']
        }

        node_specs['PD_and_T1w'] = {
            'func': nodes.PD_and_T1w_func,
            'parents': ['sequence_type', 'TR', 'TI', 'FA', 'field_strength', 'tissues']
        }

        node_specs['reference_signal'] = {
            'func': nodes.reference_signal_func,
            'parents': ['decayed_signal', 'PD_and_T1w', 'reference_tissue']
        }

        node_specs['noise_std'] = {
            'func': nodes.noise_std_func,
            'parents': ['sampling_time', 'noise_gain', 'NSA', 'field_strength']
        }

        node_specs['noise'] = {
            'func': nodes.noise_func,
            'parents': ['k_samples', 'noise_std']
        }

        node_specs['SNR'] = {
            'func': nodes.SNR_func,
            'parents': ['reference_signal', 'noise_std']
        }

        node_specs['relative_SNR'] = {
            'func': nodes.relative_SNR_func,
            'parents': ['SNR', 'reference_SNR']
        }
        
        node_specs['FOV_box'] = {
            'func': nodes.FOV_box_func,
            'parents': ['show_FOV', 'is_radial', 'FOV', 'matrix', 'freq_dir', 'phase_dir', 'k_read_axis', 'k_phase_axis']
        }
        
        node_specs['scantime'] = {
            'func': nodes.scantime_func,
            'parents': ['num_shots', 'NSA', 'TR']
        }
        
        node_specs['measured_kspace'] = {
            'func': nodes.measured_kspace_func,
            'parents': ['noise', 'kspace_comps', 'FatSat', 'PD_and_T1w']
        }
        
        node_specs['gridded_kspace'] = {
            'func': nodes.gridded_kspace_func,
            'parents': ['k_grid_axes', 'is_radial', 'measured_kspace', 'k_samples', 'FOV', 'matrix']
        }
        
        node_specs['full_kspace'] = {
            'func': nodes.full_kspace_func,
            'parents': ['num_blank_lines', 'is_radial', 'gridded_kspace', 'phase_dir', 'homodyne', 'k_phase_axis']
        }
        
        node_specs['full_k_matrix'] = {
            'func': nodes.full_k_matrix_func, 
            'parents': ['full_kspace']
        }
        
        node_specs['apodized_kspace'] = {
            'func': nodes.apodized_kspace_func,
            'parents': ['full_kspace', 'do_apodize', 'apodization_alpha']
        }
        
        node_specs['oversampled_recon_matrix'] = {
            'func': nodes.oversampled_recon_matrix_func,
            'parents': ['recon_matrix', 'full_k_matrix', 'matrix']
        }
        
        node_specs['k_trajectory'] = {
            'func': nodes.k_trajectory_func,
            'parents': ['RF_refocusing', 'frequency_board', 'phase_board', 'is_radial', 'phase_dir', 'spoke_angle']
        }

        node_specs['time_dim'] = {
            'func': nodes.time_dim_func,
            'parents': []
        }

        node_specs['frequency_dim'] = {
            'func': nodes.frequency_dim_func,
            'parents': []
        }

        node_specs['phase_dim'] = {
            'func': nodes.phase_dim_func,
            'parents': []
        }

        node_specs['slice_dim'] = {
            'func': nodes.slice_dim_func,
            'parents': []
        }

        node_specs['RF_dim'] = {
            'func': nodes.RF_dim_func,
            'parents': []
        }

        node_specs['signal_dim'] = {
            'func': nodes.signal_dim_func,
            'parents': []
        }

        node_specs['ADC_dim'] = {
            'func': nodes.ADC_dim_func,
            'parents': []
        }

        node_specs['frequency_objects'] = {
            'func': nodes.frequency_objects_func,
            'parents': ['read_prephaser', 'readouts']
        }

        node_specs['phase_objects'] = {
            'func': nodes.phase_objects_func,
            'parents': ['phasers', 'rephasers', 'blips']
        }
        
        node_specs['slice_objects'] = {
            'func': nodes.slice_objects_func,
            'parents': ['slice_select_inversion', 'inversion_spoiler', 'FatSat_spoiler', 'slice_select_excitation', 'slice_select_rephaser', 'slice_select_refocusing', 'spoiler']
        }
        
        node_specs['RF_objects'] = {
            'func': nodes.RF_objects_func,
            'parents': ['RF_inversion', 'RF_FatSat', 'RF_excitation', 'RF_refocusing']
        }
        
        node_specs['signal_objects'] = {
            'func': nodes.signal_objects_func,
            'parents': ['signal_curves']
        }
        
        node_specs['ADC_objects'] = {
            'func': nodes.ADC_objects_func,
            'parents': ['sampling_windows']
        }
        
        node_specs['TR_span'] = {
            'func': nodes.TR_span_func,
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
            'func': nodes.frequency_board_func,
            'parents': ['time_dim', 'frequency_dim', 'frequency_objects', 'TR_span', 'frequency_hover']
        }

        node_specs['phase_board'] = {
            'func': nodes.phase_board_func,
            'parents': ['time_dim', 'phase_dim', 'phase_objects', 'TR_span', 'phase_hover']
        }

        node_specs['slice_board'] = {
            'func': nodes.slice_board_func,
            'parents': ['time_dim', 'slice_dim', 'slice_objects', 'TR_span', 'slice_hover']
        }

        node_specs['RF_board'] = {
            'func': nodes.RF_board_func,
            'parents': ['time_dim', 'RF_dim', 'RF_objects', 'TR_span', 'RF_hover']
        }

        node_specs['signal_board'] = {
            'func': nodes.signal_board_func,
            'parents': ['time_dim', 'signal_dim', 'signal_objects', 'ADC_objects', 'TR_span', 'signal_hover']
        }

        node_specs['sequence_plot'] = {
            'func': nodes.sequence_plot_func,
            'parents': ['frequency_board', 'phase_board', 'slice_board', 'RF_board', 'signal_board']
        }
        
        node_specs['kspace'] = {
            'func': nodes.kspace_func,
            'parents': ['kspace_type', 'show_processed_kspace', 'oversampled_recon_matrix', 'FOV', 'recon_matrix', 'full_k_matrix', 'zerofilled_kspace', 'kspace_exponent', 'gridded_kspace', 'k_grid_axes']
        }
        
        node_specs['zerofilled_kspace'] = {
            'func': nodes.zerofilled_kspace_func,
            'parents': ['apodized_kspace', 'oversampled_recon_matrix']
        }
        
        node_specs['image_array'] = {
            'func': nodes.image_array_func,
            'parents': ['oversampled_recon_matrix', 'full_k_matrix', 'recon_matrix', 'zerofilled_kspace']
        }
        
        node_specs['image'] = {
            'func': nodes.image_func,
            'parents': ['image_type', 'recon_matrix', 'FOV', 'image_array']
        }

        self.graph = Graph(node_specs)

        for node in self.graph.nodes.values():
            # add watchers for input nodes
            if node.name in self.param and not node.parents:
                def on_change(node, graph, event):
                    node.invalidate()
                    graph.flush_actions()
                self.param.watch(partial(on_change, node, self.graph), node.name)

        self.set_reference_SNR()

        self.derived_params = ['FOV_bandwidth', 'FW_shift', 'SNR', 'name', 'num_shots', 'recon_voxel_F', 'recon_voxel_P', 'reference_SNR', 'relative_SNR', 'scantime', 'spoke_angle', 'voxel_F', 'voxel_P', 'shot_label']

    def init_bounds(self):
        self.param.object.objects = phantom.get_phantom_names()
        self.param.field_strength.objects=[1.5, 3.0]
        self.param.parameter_style.objects=['Matrix and Pixel BW', 'Voxel_size and Fat/water shift', 'Matrix and FOV BW']
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
        for par, values in constants.PARAM_VALUES.items():
            self.param[par].objects=values

    def get_params(self):
        return {param: self.__getattribute__(param) for param in self.param.values().keys() if param not in self.derived_params}

    def set_params(self, settings):
        self.init_bounds()
        self.param.update(settings)

    def set_param_discrete_bounds(self, par, curval, minval=None, maxval=None):
        values = constants.PARAM_VALUES[par.name]
        vals = values.values() if isinstance(values, dict) else values
        if minval is None:
            minval = -np.inf
        if maxval is None:
            maxval = np.inf
        minval = min([v for v in vals if v >= minval], default=minval)
        maxval = max([v for v in vals if v <= maxval], default=maxval)

        outbound = False
        if curval < minval:
            warnings.warn(f'trying to set {par.name} min bound above current value ({minval} > {curval})')
            outbound = True
        if curval > maxval:
            warnings.warn(f'trying to set {par.name} max bound below current value ({maxval} < {curval})')
            outbound = True
        if outbound:
            return self.handle_outbound(par.name)
        
        objects = {k: v for k, v in values.items() if minval <= v <= maxval} if isinstance(values, dict) else [v for v in values if minval <= v <= maxval]
        par.objects = objects

    def set_param_bounds(self, par, minval=None, maxval=None):
        if isinstance(minval, list):
            minval = max(minval) if minval else None
        if isinstance(maxval, list):
            maxval = min(maxval) if maxval else None
        curval = getattr(self, par.name)
        if type(par) is param.parameters.Selector:
            return self.set_param_discrete_bounds(par, curval, minval, maxval)
        
        outbound = False
        if curval < minval:
            warnings.warn(f'trying to set {par.name} min bound above current value ({minval} > {curval})')
            outbound = True
        if curval > maxval:
            warnings.warn(f'trying to set {par.name} max bound below current value ({maxval} < {curval})')
            outbound = True
        if outbound:
            return self.handle_outbound(par.name)
        par.bounds = (minval, maxval)

    def _watch_FOV_F(self):
        with param.parameterized.batch_call_watchers(self):
            if self.parameter_style=='Voxel_size and Fat/water shift' or self.is_radial:
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
            if self.parameter_style=='Voxel_size and Fat/water shift' or self.is_radial:
                self.set_closest(self.param.matrix_P, self.FOV_P/self.voxel_P)
                self.set_closest(self.param.recon_matrix_P, self.FOV_P/self.recon_voxel_P)
            self.update_voxel_P_objects()
            self.update_recon_voxel_P_objects()
            self.set_closest(self.param.voxel_P, self.FOV_P/self.matrix_P)
            self.set_closest(self.param.recon_voxel_P, self.FOV_P/self.recon_matrix_P)
            self.param.trigger('voxel_P', 'recon_voxel_P')

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
            self.update_recon_voxel_F_objects()
            self.set_closest(self.param.recon_matrix_F, self.matrix_F * self.rec_acq_ratio_F)
            self.param.trigger('voxel_F', 'recon_voxel_F')
            if self.is_radial:
                self.set_closest(self.param.matrix_P, self.matrix_F * self.FOV_P / self.FOV_F)

    def _watch_matrix_P(self):
        with param.parameterized.batch_call_watchers(self):
            self.update_voxel_P_objects()
            self.set_closest(self.param.voxel_P, self.FOV_P / self.matrix_P)
            self.update_recon_voxel_P_objects()
            self.set_closest(self.param.recon_matrix_P, self.matrix_P * self.rec_acq_ratio_P)
            self.param.trigger('voxel_P', 'recon_voxel_P')
            if self.is_radial:
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

    def _watch_field_strength(self):
        with param.parameterized.batch_call_watchers(self):
            self.update_FW_shift_objects()
            self.set_closest(self.param.FW_shift, pixel_BW_to_shift(self.pixel_bandwidth, self.field_strength))
            self.param.trigger('FW_shift')

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

    def _watch_recon_matrix_F(self):
        self.set_closest(self.param.recon_voxel_F, self.FOV_F / self.recon_matrix_F)

    def _watch_recon_matrix_P(self):
        self.set_closest(self.param.recon_voxel_P, self.FOV_P / self.recon_matrix_P)

    def set_visibility(self, par_name, visible):
        precedence = abs(self.param[par_name].precedence)
        if not visible:
            precedence *= -1
        self.param[par_name].precedence = precedence
    
    def set_param(self, par, value=None, values=None, mode='round'):
        # par.objects could be dict or param.ListProxy
        if not values:
            if hasattr(par, 'objects'):
                values = par.objects
                if not values:
                    warnings.warn(f'Could not set {par.name} since no allowed values')
                    return
        if callable(getattr(values, 'values', False)):
            values = values.values()
        current = getattr(self, par.name)
        if value is None:
            value = current
        if values:
            match mode:
                case 'round':
                    new = min(values, key=lambda x: abs(x-value))
                case 'ceil':
                    new = min([v for v in values if v >= value])
                case 'floor':
                    new = max([v for v in values if v <= value])
                case _:
                    raise ValueError(f'Invalid mode {mode}')
        else:
            new = value
        if new != current:
            setattr(self, par.name, new)

    def handle_outbound(self, par_name):
        bounds_func = f'set_{par_name}_bounds'
        self.graph.nodes[bounds_func]._queued = False
        self.graph.nodes[bounds_func].invalidate()
        min_TE = self.graph.nodes['min_TE'].value
        if self.TE < min_TE:
            print(f'Increasing TE from {self.TE} to {min_TE} to resolve conflict')
            self.set_param(self.param.TE, min_TE, values=constants.PARAM_VALUES['TE'], mode='ceil')
            return
        min_TR =  self.graph.nodes['min_TR'].value
        if self.TR < min_TR:
            print(f'Increasing TR from {self.TR} to {min_TR} to resolve conflict')
            self.set_param(self.param.TR, min_TR, mode='ceil')
            return
        raise NotImplementedError(f'Could not resolve conflict for outbound parameter {par_name}')
    
    def set_isotropic_voxel_size(self, is_radial, FOV_F, matrix_F, FOV_P, matrix_P):
        if is_radial:
            if (FOV_F / matrix_F < FOV_P / matrix_P):
                self.set_param(self.param.matrix_P, matrix_F * FOV_P / FOV_F, mode='round')
            else:
                self.set_param(self.param.matrix_F, matrix_P * FOV_F / FOV_P, mode='round')

    def set_TR_bounds(self, min_TR):
        self.set_param_bounds(self.param.TR, minval=min_TR)

    def set_TE_bounds(self, min_TE, max_TE):
        self.set_param_bounds(self.param.TE, minval=min_TE, maxval=max_TE)

    def set_TI_bounds(self, sequence_type, TR, spoiler, slice_select_inversion_floating):
        if sequence_type != 'Inversion Recovery':
            return
        max_TI = TR - spoiler['time'][-1] - slice_select_inversion_floating['dur_f'] / 2
        max_TI = max([v for v in constants.PARAM_VALUES['TI'].values() if v <= max_TI])
        self.set_param_bounds(self.param.TI, maxval=max_TI)

    def set_x_y_labels(self, frequency_direction):
        for p in [self.param.FOV_F, self.param.FOV_P, self.param.matrix_F, self.param.matrix_P, self.param.recon_matrix_F, self.param.recon_matrix_P]:
            if (' y' in p.label) and (('_F' in p.name and frequency_direction=='left-right') or
                                        ('_P' in p.name and frequency_direction=='anterior-posterior')):
                p.label = p.label.replace(' y', ' x')
            elif (' x' in p.label) and (('_P' in p.name and frequency_direction=='left-right') or
                                        ('_F' in p.name and frequency_direction=='anterior-posterior')):
                p.label = p.label.replace(' x', ' y')

    def set_labels_by_trajectory(self, shot_label):
        self.param.shot.label = f'Displayed {shot_label}'
        self.param.radial_factor.label = f'{shot_label.capitalize()} sampling factor'

    def set_shot_label(self, shot_label):
        self.set_param(self.param.shot_label, shot_label)
    
    def set_spoke_angle(self, spoke_angle):
        self.set_param(self.param.spoke_angle, spoke_angle)
    
    def set_num_shots(self, num_shots):
        self.set_param(self.param.num_shots, num_shots)
    
    def set_relative_SNR(self, relative_SNR):
        self.set_param(self.param.relative_SNR, relative_SNR)
    
    def set_scantime(self, scantime):
        self.set_param(self.param.scantime, scantime)
    
    def set_trajectory_objects(self, EPI_factor, turbo_factor):
        # Label radial trajectory 'Radial' or 'PROPELLER' depending on nLines per shot
        self.param.trajectory.objects = constants.TRAJECTORIES
        invalid, updated = ('PROPELLER', 'Radial') if (EPI_factor * turbo_factor == 1) else ('Radial', 'PROPELLER')
        if self.trajectory == invalid:
            self.trajectory = updated
        self.param.trajectory.objects = [t for t in constants.TRAJECTORIES if t != invalid]

    def set_pixel_bandwidth_bounds(self, matrix_F, FOV_F, is_gradient_echo, RF_refocusing, turbo_factor, EPI_factor, TE, RF_excitation, refocusing_time, readtrain_spacing, TR, sequence_start, spoiler, k0_echo_indices_linear_order, k0_echo_indices_reverse_linear_order, reverse_linear_order, read_prephaser, phaser_duration, slice_select_excitation, slice_select_rephaser, max_blip_dur, slice_select_refocusing):
        readout_area = 1e3 * matrix_F / (FOV_F * constants.GYRO)
        # min limit imposed by maximum gradient amplitude:
        min_read_duration = readout_area / constants.MAX_AMP

        first_readtrain_ref = TE if (turbo_factor == 1) else readtrain_spacing
        last_readtrain_ref = TE if (turbo_factor == 1) else readtrain_spacing * turbo_factor
        last_read_end = TR + sequence_start - spoiler['dur_f']
        max_dur_ref_to_read_end = last_read_end - last_readtrain_ref
        
        slice_select_rewind_dur = slice_select_excitation['risetime_f'] + slice_select_rephaser['dur_f'] if is_gradient_echo else slice_select_refocusing[0]['risetime_f']

        k0_echo_indices = k0_echo_indices_linear_order if reverse_linear_order else k0_echo_indices_reverse_linear_order
        
        pixel_bandwidth_values = []
        for pixel_bandwidth in constants.PARAM_VALUES['pixel_bandwidth'].values():
            
            read_duration = 1e3 / pixel_bandwidth
            
            if read_duration < min_read_duration:
                continue
            
            readout_risetime = readout_area / read_duration / constants.MAX_SLEW
            readout_gap = max(max_blip_dur, 2 * readout_risetime)
            
            for (gr_echo, _) in k0_echo_indices:
                num_blips_before_ref = gr_echo if (turbo_factor == 1) else (EPI_factor-1) / 2
                num_blips_after_ref = EPI_factor - 1 - num_blips_before_ref
                
                dur_ref_to_read_end = read_duration * (num_blips_after_ref + 1/2) + readout_gap * num_blips_after_ref
                if dur_ref_to_read_end > max_dur_ref_to_read_end:
                    continue
                
                if is_gradient_echo:
                    first_read_start = RF_excitation['dur_f']/2 + max(
                        read_prephaser['dur_f'] + readout_risetime,
                        phaser_duration,
                        slice_select_rewind_dur
                        )
                else: # spin echo
                    refocusing_dur = RF_refocusing[0]['dur_f']
                    first_read_start = refocusing_time[0] + refocusing_dur / 2 + max(
                        readout_risetime,
                        phaser_duration,
                        slice_select_rewind_dur
                        )
                
                max_dur_read_start_to_ref = first_readtrain_ref - first_read_start
                dur_read_start_to_ref = read_duration * (num_blips_before_ref + 1/2) + readout_gap * num_blips_before_ref
                
                if dur_read_start_to_ref > max_dur_read_start_to_ref:
                    continue

                pixel_bandwidth_values.append(pixel_bandwidth)

        min_pixel_bandwidth = min(pixel_bandwidth_values, default=np.inf)
        max_pixel_bandwidth = max(pixel_bandwidth_values, default=-np.inf)
        self.set_param_bounds(self.param.pixel_bandwidth, minval=min_pixel_bandwidth, maxval=max_pixel_bandwidth)

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

    def set_recon_matrix_F_bounds(self, matrix_F):
        self.set_param_bounds(self.param['recon_matrix_F'], minval=matrix_F)

    def set_recon_matrix_P_bounds(self, matrix_P):
        self.set_param_bounds(self.param['recon_matrix_P'], minval=matrix_P)

    def set_FOV_F_bounds(self, matrix_F, max_readout_area, parameter_style, voxel_F, recon_voxel_F):
        min_FOV = [1e3 * matrix_F / (max_readout_area * constants.GYRO) if max_readout_area > 0 else np.inf]
        max_FOV = []
        if parameter_style == 'Voxel_size and Fat/water shift':
            # TODO: this could be solved better
            min_FOV.append(voxel_F * self.param.matrix_F.objects[0])
            min_FOV.append(recon_voxel_F * self.param.recon_matrix_F.objects[0])
            max_FOV.append(voxel_F * self.param.matrix_F.objects[-1])
            max_FOV.append(recon_voxel_F * self.param.recon_matrix_F.objects[-1])
        self.set_param_bounds(self.param.FOV_F, minval=min_FOV, maxval=max_FOV)

    def set_FOV_P_bounds(self, matrix_P, max_phaser_area, parameter_style, voxel_P, recon_voxel_P):
        min_FOV = [(matrix_P - 1) / (max_phaser_area * constants.GYRO * 2e-3)]
        max_FOV = []
        if parameter_style == 'Voxel_size and Fat/water shift':
            # TODO: this could be solved better
            min_FOV.append(voxel_P * self.param.matrix_P.objects[0])
            min_FOV.append(recon_voxel_P * self.param.recon_matrix_P.objects[0])
            max_FOV.append(voxel_P * self.param.matrix_P.objects[-1])
            max_FOV.append(recon_voxel_P * self.param.recon_matrix_P.objects[-1])
        self.set_param_bounds(self.param.FOV_P, minval=min_FOV, maxval=max_FOV)

    def set_FW_shift_objects(self, field_strength):
        self.param.FW_shift.objects = {f'{formatting.format_float(shift, 2)} pixels': shift for shift in [pixel_BW_to_shift(pBW, field_strength) for pBW in list(self.param.pixel_bandwidth.objects.values())[::-1]]}

    def set_FOV_bandwidth_objects(self, matrix_F):
        self.param.FOV_bandwidth.objects = {f'±{formatting.format_float(bw, 3)} kHz': bw for bw in [pixel_BW_to_FOV_BW(pBW, matrix_F) for pBW in self.param.pixel_bandwidth.objects.values()]}

    def set_voxel_F_objects(self, FOV_F):
        self.param.voxel_F.objects = {f'{formatting.format_float(voxel, 3)} mm': voxel for voxel in [FOV_F / matrix for matrix in self.param.matrix_F.objects[::-1]]}

    def set_voxel_P_objects(self, FOV_P):
        self.param.voxel_P.objects = {f'{formatting.format_float(voxel, 3)} mm': voxel for voxel in [FOV_P / matrix for matrix in self.param.matrix_P.objects[::-1]]}

    def set_recon_voxel_F_objects(self, FOV_F):
        self.param.recon_voxel_F.objects = {f'{formatting.format_float(voxel, 3)} mm': voxel for voxel in [FOV_F / matrix for matrix in self.param.recon_matrix_F.objects[::-1]]}

    def set_recon_voxel_P_objects(self, FOV_P):
        self.param.recon_voxel_P.objects = {f'{formatting.format_float(voxel, 3)} mm': voxel for voxel in [FOV_P / matrix for matrix in self.param.recon_matrix_P.objects[::-1]]}

    def set_slice_thickness_bounds(self, RF_excitation, is_gradient_echo, RF_refocusing, sequence_type, RF_inversion, TR, spoiler, sampling_windows):
        min_thks = [RF_excitation['FWHM_f'] / (constants.MAX_AMP * constants.GYRO)]
        if not is_gradient_echo:
            min_thks.append(RF_refocusing[0]['FWHM_f'] / (constants.MAX_AMP * constants.GYRO))
        if sequence_type == 'Inversion Recovery':
            min_thks.append(RF_inversion['FWHM_f'] / (constants.MAX_AMP * constants.GYRO) * constants.INVERSION_THK_FACTOR)
        
        # Constraint due to TR: 
        if sequence_type == 'Inversion Recovery':
            max_risetime = TR - (spoiler['time'][-1] - RF_inversion['time'][0])
            max_amp = constants.MAX_SLEW * max_risetime
            min_thks.append(RF_inversion['FWHM_f'] / (max_amp * constants.GYRO))
        else:
            max_risetime = TR - (spoiler['time'][-1] - RF_excitation['time'][0])
            max_amp = constants.MAX_SLEW * max_risetime
            min_thks.append(RF_excitation['FWHM_f'] / (max_amp * constants.GYRO))
        
        # See paramBounds.tex for formulae
        s = constants.MAX_SLEW
        d = RF_excitation['dur_f']
        if is_gradient_echo: # Constraint due to slice rephaser
            t = sampling_windows[0][0]['time'][0]
            h = s * (t - np.sqrt(t**2/2 + d**2/8))
            h = min(h, constants.MAX_AMP)
            A = d * (np.sqrt((d*s+2*h)**2 - 8*h*(h-s*(t-d/2))) - d*s - 2*h) / 2
        else: # Spin echo: Constraint due to slice rephaser and refocusing slice select rampup
            t = RF_refocusing[0]['time'][0]
            h = s * (np.sqrt(2*(d + 2*t)**2 - 4*d**2) - d - 2*t) / 4
            h = min(h, constants.MAX_AMP)
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

    def set_reference_tissue_objects(self, tissues):
        self.param.reference_tissue.objects = tissues
        self.reference_tissue = tissues[0]

    def set_shot_bounds(self, num_shots):
        self.param.shot.bounds = (1, num_shots)
        self.shot = min(self.shot, num_shots)

    def set_parameter_style_visibility(self, parameter_style):
        self.set_visibility('pixel_bandwidth', parameter_style == 'Matrix and Pixel BW')
        self.set_visibility('FOV_bandwidth', parameter_style == 'Matrix and FOV BW')
        self.set_visibility('FW_shift', parameter_style == 'Voxel_size and Fat/water shift')
        for voxel_size_param in ['voxel_F', 'voxel_P', 'recon_voxel_F', 'recon_voxel_P']:
            self.set_visibility(voxel_size_param, parameter_style == 'Voxel_size and Fat/water shift')
        for matrix_param in ['matrix_F', 'matrix_P']:
            self.set_visibility(matrix_param, parameter_style != 'Voxel_size and Fat/water shift')

    def set_partial_Fourier_visibility(self, is_radial):
        visible = not is_radial
        self.set_visibility('partial_Fourier', visible)
        if not visible:
            self.set_param(self.param.partial_Fourier, 1)

    def set_frequency_direction_visibility(self, is_radial):
        self.set_visibility('frequency_direction', not is_radial)

    def set_phase_oversampling_visibility(self, is_radial):
        visible = not is_radial
        self.set_visibility('phase_oversampling', visible)
        if not visible:
            self.set_param(self.param.phase_oversampling, 0)

    def set_radial_factor_visibility(self, is_radial):
        self.set_visibility('radial_factor', is_radial)

    def set_TI_visibility(self, sequence_type):
        self.set_visibility('TI', sequence_type == 'Inversion Recovery')

    def set_FA_visibility(self, sequence_type):
        self.set_visibility('FA', sequence_type == 'Spoiled Gradient Echo')

    def set_turbo_factor_visibility(self, sequence_type):
        visible = sequence_type != 'Spoiled Gradient Echo'
        self.set_visibility('turbo_factor', visible)
        if not visible:
            self.set_param(self.param.turbo_factor, 1)

    def set_homodyne_visibility(self, num_blank_lines, is_radial):
        self.set_visibility('homodyne', (num_blank_lines > 0 and not is_radial))

    def set_apodization_alpha_visibility(self, do_apodize):
        self.set_visibility('apodization_alpha', do_apodize)

    def get_hover_tool(self, board, attributes):
        with open(Path(__file__).parent / 'hoverCallback.js', 'r') as file:
            hover_callback = CustomJS(args={'hover_index': self.hover_index, 'board': board}, code=file.read())
        if board == 'slice':
            hover_callback = None
        return HoverTool(tooltips=[(attr, f'@{attr}') for attr in attributes], attachment='below', callback=hover_callback)

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
    
    def update_k_line_coords(self, attr, old, hover_index):
        if len(hover_index['index']) == 0:
            self.k_line.event(coords=[None])
            return
        board = hover_index['board'][0]
        index = hover_index['index'][0]
        object = self.graph.nodes[f'{board}_objects'].value[index]
        k_trajectory = self.graph.nodes['k_trajectory'].value
        self.k_line.event(coords=list(get_k_on_interval(object['time'][[0, -1]], k_trajectory)))
    
    def set_reference_SNR(self, event=None):
        self.reference_SNR = self.graph.nodes['SNR'].value
    
    @param.depends('sequence_type', 'FatSat', 'TR', 'TE', 'FA', 'TI', 'FOV_F', 'FOV_P', 'phase_oversampling', 'num_shots', 'matrix_F', 'matrix_P', 'slice_thickness', 'trajectory', 'frequency_direction', 'pixel_bandwidth', 'partial_Fourier', 'turbo_factor', 'EPI_factor', 'shot')
    def display_sequence_plot(self):
        return self.graph.nodes['sequence_plot'].value
    
    @param.depends('object', 'field_strength', 'sequence_type', 'FatSat', 'TR', 'TE', 'FA', 'TI', 'FOV_F', 'FOV_P', 'phase_oversampling', 'num_shots', 'matrix_F', 'matrix_P', 'recon_matrix_F', 'recon_matrix_P', 'slice_thickness', 'trajectory', 'frequency_direction', 'pixel_bandwidth', 'NSA', 'partial_Fourier', 'turbo_factor', 'EPI_factor', 'kspace_type', 'show_processed_kspace', 'kspace_exponent', 'homodyne', 'do_apodize', 'apodization_alpha', 'do_zerofill', 'radial_FOV_oversampling')
    def display_kspace(self):
        return self.graph.nodes['kspace'].value

    @param.depends('object', 'field_strength', 'sequence_type', 'FatSat', 'TR', 'TE', 'FA', 'TI', 'FOV_F', 'FOV_P', 'phase_oversampling', 'num_shots', 'matrix_F', 'matrix_P', 'recon_matrix_F', 'recon_matrix_P', 'slice_thickness', 'trajectory', 'frequency_direction', 'pixel_bandwidth', 'NSA', 'partial_Fourier', 'turbo_factor', 'EPI_factor', 'image_type', 'show_FOV', 'homodyne', 'do_apodize', 'apodization_alpha', 'do_zerofill', 'radial_FOV_oversampling')
    def display_image(self):
        return self.graph.nodes['image'].value * self.graph.nodes['FOV_box'].value