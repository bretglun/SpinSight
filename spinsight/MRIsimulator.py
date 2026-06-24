import holoviews as hv
from holoviews import streams
import param
import numpy as np
import math
from pathlib import Path
from spinsight import constants, convert, formatting, phantom
from spinsight import nodes # needed to initialize graph node decorators
from spinsight.DAG import Graph
from bokeh.models import HoverTool, CustomJS, ColumnDataSource
import warnings

hv.extension('bokeh')


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
    matrix_P_param = param.Selector(default=180, precedence=4, label='Acquisition matrix x')
    matrix_F_param = param.Selector(default=180, precedence=4, label='Acquisition matrix y')
    recon_voxel_P = param.Selector(default=0.666, precedence=-5, label='Reconstructed voxel size x')
    recon_voxel_F = param.Selector(default=0.666, precedence=-5, label='Reconstructed voxel size y')
    recon_matrix_P_param = param.Selector(default=360, precedence=5, label='Reconstruction matrix x')
    recon_matrix_F_param = param.Selector(default=360, precedence=5, label='Reconstruction matrix y')
    slice_thickness = param.Selector(default=3, precedence=6, label='Slice thickness')
    radial_FOV_oversampling = param.Number(default=2, step=0.01, precedence=9, label='Radial FOV oversampling factor')
    
    sequence_type = param.ObjectSelector(default=constants.SEQUENCES[0], precedence=1, label='Pulse sequence')
    pixel_bandwidth_param = param.Selector(default=480, precedence=2, label='Pixel bandwidth')
    FOV_bandwidth = param.Selector(default=convert.pixel_BW_to_FOV_BW(480, 180), precedence=-2, label='FOV bandwidth')
    FW_shift = param.Selector(default=convert.pixel_BW_to_shift(480), precedence=-2, label='Fat/water shift')
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

        self.derived_params = {'FOV_bandwidth', 'FW_shift', 'SNR', 'name', 'num_shots', 'recon_voxel_F', 'recon_voxel_P', 'relative_SNR', 'scantime', 'spoke_angle', 'voxel_F', 'voxel_P', 'shot_label'}

        self.graph = Graph(self)

        self.set_reference_SNR()

    def init_bounds(self):
        self.param.object.objects = phantom.get_phantom_names()
        self.param.field_strength.objects=[1.5, 3.0]
        self.param.parameter_style.objects=constants.PARAMETER_STYLES
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
        vals = objects.values() if isinstance(objects, dict) else objects

        if curval not in vals:
            if isinstance(objects, dict):
                cur_label = next((k for k, v in par.names.items() if v==curval), str(curval))
                objects[cur_label] = curval
                objects = dict(sorted(objects.items()))
            else:
                objects = sorted(objects.append(curval))
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
    
    @Graph.node(action_precedence=1, simulator=True)
    def set_isotropic_voxel_size(self, is_radial, FOV_F, matrix_F, FOV_P, matrix_P):
        # TODO: needs repair, for instance set param according to param_style
        if is_radial:
            if (FOV_F / matrix_F < FOV_P / matrix_P):
                self.set_param(self.param.matrix_P, matrix_F * FOV_P / FOV_F, mode='round')
            else:
                self.set_param(self.param.matrix_F, matrix_P * FOV_F / FOV_P, mode='round')

    @Graph.node(action_precedence=1, simulator=True)
    def set_TR_bounds(self, min_TR):
        self.set_param_bounds(self.param.TR, minval=min_TR)

    @Graph.node(action_precedence=1, simulator=True)
    def set_TE_bounds(self, min_TE, max_TE):
        self.set_param_bounds(self.param.TE, minval=min_TE, maxval=max_TE)

    @Graph.node(action_precedence=1, simulator=True)
    def set_TI_bounds(self, sequence_type, TR, spoiler, slice_select_inversion_floating):
        if sequence_type != 'Inversion Recovery':
            return
        max_TI = TR - spoiler['time'][-1] - slice_select_inversion_floating['dur_f'] / 2
        max_TI = max([v for v in constants.PARAM_VALUES['TI'].values() if v <= max_TI])
        self.set_param_bounds(self.param.TI, maxval=max_TI)

    @Graph.node(action_precedence=1, simulator=True)
    def set_x_y_labels(self, frequency_direction):
        for p in [self.param.FOV_F, self.param.FOV_P, self.param.matrix_F_param, self.param.matrix_P_param, self.param.recon_matrix_F_param, self.param.recon_matrix_P_param]:
            if (' y' in p.label) and (('_F' in p.name and frequency_direction=='left-right') or
                                        ('_P' in p.name and frequency_direction=='anterior-posterior')):
                p.label = p.label.replace(' y', ' x')
            elif (' x' in p.label) and (('_P' in p.name and frequency_direction=='left-right') or
                                        ('_F' in p.name and frequency_direction=='anterior-posterior')):
                p.label = p.label.replace(' x', ' y')

    @Graph.node(action_precedence=1, simulator=True)
    def set_labels_by_trajectory(self, shot_label):
        self.param.shot.label = f'Displayed {shot_label}'
        self.param.radial_factor.label = f'{shot_label.capitalize()} sampling factor'

    @Graph.node(action_precedence=1, simulator=True)
    def set_shot_label(self, shot_label):
        self.set_param(self.param.shot_label, shot_label)
    
    @Graph.node(action_precedence=1, simulator=True)
    def set_spoke_angle(self, spoke_angle):
        self.set_param(self.param.spoke_angle, spoke_angle)
    
    @Graph.node(action_precedence=1, simulator=True)
    def set_num_shots(self, num_shots):
        self.set_param(self.param.num_shots, num_shots)
    
    @Graph.node(action_precedence=1, simulator=True)
    def set_relative_SNR(self, relative_SNR):
        self.set_param(self.param.relative_SNR, relative_SNR)
    
    @Graph.node(action_precedence=1, simulator=True)
    def set_scantime(self, scantime):
        self.set_param(self.param.scantime, scantime)
    
    @Graph.node(action_precedence=1, simulator=True)
    def set_pixel_bandwidth(self, pixel_BW_is_input, pixel_bandwidth):
        if not pixel_BW_is_input:
            self.set_param(self.param.pixel_bandwidth_param, pixel_bandwidth)
    
    @Graph.node(action_precedence=1, simulator=True)
    def set_FOV_bandwidth(self, FOV_BW_is_input, pixel_bandwidth, matrix_F):
        if not FOV_BW_is_input:
            self.set_param(self.param.FOV_bandwidth, convert.pixel_BW_to_FOV_BW(pixel_bandwidth, matrix_F))
    
    @Graph.node(action_precedence=1, simulator=True)
    def set_FW_shift(self, FW_shift_is_input, pixel_bandwidth, field_strength):
        if not FW_shift_is_input:
            self.set_param(self.param.FW_shift, convert.pixel_BW_to_shift(pixel_bandwidth, field_strength))
    
    @Graph.node(action_precedence=1, simulator=True)
    def set_matrix_F(self, matrix_is_input, matrix_F):
        if not matrix_is_input:
            self.set_param(self.param.matrix_F_param, matrix_F)

    @Graph.node(action_precedence=1, simulator=True)
    def set_matrix_P(self, matrix_is_input, matrix_P):
        if not matrix_is_input:
            self.set_param(self.param.matrix_P_param, matrix_P)

    @Graph.node(action_precedence=1, simulator=True)
    def set_recon_matrix_F(self, matrix_is_input, recon_matrix_F):
        if not matrix_is_input:
            self.set_param(self.param.recon_matrix_F_param, recon_matrix_F)

    @Graph.node(action_precedence=1, simulator=True)
    def set_recon_matrix_P(self, matrix_is_input, recon_matrix_P):
        if not matrix_is_input:
            self.set_param(self.param.recon_matrix_P_param, recon_matrix_P)

    @Graph.node(action_precedence=1, simulator=True)
    def set_voxel_F(self, voxel_size_is_input, FOV_F, matrix_F):
        if not voxel_size_is_input:
            self.set_param(self.param.voxel_F, FOV_F / matrix_F)

    @Graph.node(action_precedence=1, simulator=True)
    def set_voxel_P(self, voxel_size_is_input, FOV_P, matrix_P):
        if not voxel_size_is_input:
            self.set_param(self.param.voxel_P, FOV_P / matrix_P)

    @Graph.node(action_precedence=1, simulator=True)
    def set_recon_voxel_F(self, voxel_size_is_input, FOV_F, recon_matrix_F):
        if not voxel_size_is_input:
            self.set_param(self.param.recon_voxel_F, FOV_F / recon_matrix_F)

    @Graph.node(action_precedence=1, simulator=True)
    def set_recon_voxel_P(self, voxel_size_is_input, FOV_P, recon_matrix_P):
        if not voxel_size_is_input:
            self.set_param(self.param.recon_voxel_P, FOV_P / recon_matrix_P)

    @Graph.node(action_precedence=1, simulator=True)
    def set_trajectory_objects(self, EPI_factor, turbo_factor):
        # Label radial trajectory 'Radial' or 'PROPELLER' depending on nLines per shot
        self.param.trajectory.objects = constants.TRAJECTORIES
        invalid, updated = ('PROPELLER', 'Radial') if (EPI_factor * turbo_factor == 1) else ('Radial', 'PROPELLER')
        if self.trajectory == invalid:
            self.trajectory = updated
        self.param.trajectory.objects = [t for t in constants.TRAJECTORIES if t != invalid]

    @Graph.node(action_precedence=1, simulator=True)
    def set_pixel_bandwidth_bounds(self, pixel_BW_is_input, pixel_bandwidth_bounds):
        if pixel_BW_is_input:
            self.set_param_bounds(self.param.pixel_bandwidth_param, minval=pixel_bandwidth_bounds.min, maxval=pixel_bandwidth_bounds.max)

    @Graph.node(action_precedence=1, simulator=True)
    def set_FOV_bandwidth_bounds(self, FOV_BW_is_input, pixel_bandwidth_bounds, matrix_F):
        if FOV_BW_is_input:
            self.set_param_bounds(self.param.FOV_bandwidth, minval=convert.pixel_BW_to_FOV_BW(pixel_bandwidth_bounds.min, matrix_F), maxval=convert.pixel_BW_to_FOV_BW(pixel_bandwidth_bounds.max, matrix_F))
    
    @Graph.node(action_precedence=1, simulator=True)
    def set_FW_shift_bounds(self, FW_shift_is_input, pixel_bandwidth_bounds, field_strength):
        if FW_shift_is_input:
            self.set_param_bounds(self.param.FW_shift, minval=convert.pixel_BW_to_shift(pixel_bandwidth_bounds.max, field_strength), maxval=convert.pixel_BW_to_shift(pixel_bandwidth_bounds.min, field_strength))

    @Graph.node(action_precedence=1, simulator=True)
    def set_matrix_F_bounds(self, matrix_is_input, matrix_F_bounds):
        if matrix_is_input:
            self.set_param_bounds(self.param.matrix_F_param, minval=matrix_F_bounds.min, maxval=matrix_F_bounds.max)

    @Graph.node(action_precedence=1, simulator=True)
    def set_matrix_P_bounds(self, matrix_is_input, matrix_P_bounds):
        if matrix_is_input:
            self.set_param_bounds(self.param.matrix_P_param, minval=matrix_P_bounds.min, maxval=matrix_P_bounds.max)

    @Graph.node(action_precedence=1, simulator=True)
    def set_recon_matrix_F_bounds(self, matrix_is_input, recon_matrix_F_bounds):
        if matrix_is_input:
            self.set_param_bounds(self.param.recon_matrix_F_param, minval=recon_matrix_F_bounds.min, maxval=recon_matrix_F_bounds.max)

    @Graph.node(action_precedence=1, simulator=True)
    def set_recon_matrix_P_bounds(self, matrix_is_input, recon_matrix_P_bounds):
        if matrix_is_input:
            self.set_param_bounds(self.param.recon_matrix_P_param, minval=recon_matrix_P_bounds.min, maxval=recon_matrix_P_bounds.max)

    @Graph.node(action_precedence=1, simulator=True)
    def set_FOV_F_bounds(self, FOV_F_bounds):
        self.set_param_bounds(self.param.FOV_F, minval=FOV_F_bounds.min, maxval=FOV_F_bounds.max)

    @Graph.node(action_precedence=1, simulator=True)
    def set_FOV_P_bounds(self, FOV_P_bounds):
        self.set_param_bounds(self.param.FOV_P, minval=FOV_P_bounds.min, maxval=FOV_P_bounds.max)

    @Graph.node(action_precedence=1, simulator=True)
    def set_voxel_F_bounds(self, voxel_size_is_input, FOV_F, matrix_F_bounds):
        if voxel_size_is_input:
            self.set_param_bounds(self.param.voxel_F, minval=FOV_F/matrix_F_bounds.max, maxval=FOV_F/matrix_F_bounds.min)

    @Graph.node(action_precedence=1, simulator=True)
    def set_voxel_P_bounds(self, voxel_size_is_input, FOV_P, matrix_P_bounds):
        if voxel_size_is_input:
            self.set_param_bounds(self.param.voxel_P, minval=FOV_P/matrix_P_bounds.max, maxval=FOV_P/matrix_P_bounds.min)
    
    @Graph.node(action_precedence=1, simulator=True)
    def set_recon_voxel_F_bounds(self, voxel_size_is_input, FOV_F, recon_matrix_F_bounds):
        if voxel_size_is_input:
            self.set_param_bounds(self.param.recon_voxel_F, minval=FOV_F/recon_matrix_F_bounds.max, maxval=FOV_F/recon_matrix_F_bounds.min)

    @Graph.node(action_precedence=1, simulator=True)
    def set_recon_voxel_P_bounds(self, voxel_size_is_input, FOV_P, recon_matrix_P_bounds):
        if voxel_size_is_input:
            self.set_param_bounds(self.param.recon_voxel_P, minval=FOV_P/recon_matrix_P_bounds.max, maxval=FOV_P/recon_matrix_P_bounds.min)

    @Graph.node(action_precedence=1, simulator=True)
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

    @Graph.node(action_precedence=1, simulator=True)
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

    @Graph.node(action_precedence=1, simulator=True)
    def set_EPI_factor_objects(self, matrix, phase_dir, partial_Fourier, turbo_factor):
        max_EPI_factor = int(np.floor(matrix[phase_dir] * partial_Fourier / turbo_factor * 2)) # let's limit phase oversampling to 2
        self.set_param_bounds(self.param.EPI_factor, maxval=max_EPI_factor)
        # EPI_factor must be odd for turbo spin echo (GRASE)
        if self.turbo_factor > 1:
            self.param.EPI_factor.objects = [v for v in self.param.EPI_factor.objects if v%2]

    @Graph.node(action_precedence=1, simulator=True)
    def set_reference_tissue_objects(self, tissues):
        self.param.reference_tissue.objects = tissues
        self.reference_tissue = tissues[0]

    @Graph.node(action_precedence=1, simulator=True)
    def set_shot_bounds(self, num_shots):
        self.param.shot.bounds = (1, num_shots)
        self.shot = min(self.shot, num_shots)

    @Graph.node(action_precedence=1, simulator=True)
    def set_pixel_bandwidth_visibility(self, pixel_BW_is_input):
        self.set_visibility('pixel_bandwidth_param', pixel_BW_is_input)
    
    @Graph.node(action_precedence=1, simulator=True)
    def set_FOV_bandwidth_visibility(self, FOV_BW_is_input):
        self.set_visibility('FOV_bandwidth', FOV_BW_is_input)

    @Graph.node(action_precedence=1, simulator=True)
    def set_FW_shift_visibility(self, FW_shift_is_input):
        self.set_visibility('FW_shift', FW_shift_is_input)

    @Graph.node(action_precedence=1, simulator=True)
    def set_voxel_size_visibility(self, voxel_size_is_input):
        for voxel_size_param in ['voxel_F', 'voxel_P', 'recon_voxel_F', 'recon_voxel_P']:
            self.set_visibility(voxel_size_param, voxel_size_is_input)

    @Graph.node(action_precedence=1, simulator=True)
    def set_matrix_visibility(self, matrix_is_input):
        for matrix_param in ['matrix_F_param', 'matrix_P_param']:
            self.set_visibility(matrix_param, matrix_is_input)

    @Graph.node(action_precedence=1, simulator=True)
    def set_partial_Fourier_visibility(self, is_radial):
        visible = not is_radial
        self.set_visibility('partial_Fourier', visible)
        if not visible:
            self.set_param(self.param.partial_Fourier, 1)

    @Graph.node(action_precedence=1, simulator=True)
    def set_frequency_direction_visibility(self, is_radial):
        self.set_visibility('frequency_direction', not is_radial)

    @Graph.node(action_precedence=1, simulator=True)
    def set_phase_oversampling_visibility(self, is_radial):
        visible = not is_radial
        self.set_visibility('phase_oversampling', visible)
        if not visible:
            self.set_param(self.param.phase_oversampling, 0)

    @Graph.node(action_precedence=1, simulator=True)
    def set_radial_factor_visibility(self, is_radial):
        self.set_visibility('radial_factor', is_radial)

    @Graph.node(action_precedence=1, simulator=True)
    def set_TI_visibility(self, sequence_type):
        self.set_visibility('TI', sequence_type == 'Inversion Recovery')

    @Graph.node(action_precedence=1, simulator=True)
    def set_FA_visibility(self, sequence_type):
        self.set_visibility('FA', sequence_type == 'Spoiled Gradient Echo')

    @Graph.node(action_precedence=1, simulator=True)
    def set_turbo_factor_visibility(self, sequence_type):
        visible = sequence_type != 'Spoiled Gradient Echo'
        self.set_visibility('turbo_factor', visible)
        if not visible:
            self.set_param(self.param.turbo_factor, 1)

    @Graph.node(action_precedence=1, simulator=True)
    def set_homodyne_visibility(self, num_blank_lines, is_radial):
        self.set_visibility('homodyne', (num_blank_lines > 0 and not is_radial))

    @Graph.node(action_precedence=1, simulator=True)
    def set_apodization_alpha_visibility(self, do_apodize):
        self.set_visibility('apodization_alpha', do_apodize)

    def get_hover_tool(self, board, attributes):
        with open(Path(__file__).parent / 'hoverCallback.js', 'r') as file:
            hover_callback = CustomJS(args={'hover_index': self.hover_index, 'board': board}, code=file.read())
        if board == 'slice':
            hover_callback = None
        return HoverTool(tooltips=[(attr, f'@{attr}') for attr in attributes], attachment='below', callback=hover_callback)

    @Graph.node(simulator=True)
    def frequency_hover(self):
        return self.get_hover_tool('frequency', ['name', 'center', 'duration', 'area'])

    @Graph.node(simulator=True)
    def phase_hover(self):
        return self.get_hover_tool('phase', ['name', 'center', 'duration', 'area'])

    @Graph.node(simulator=True)
    def slice_hover(self):
        return self.get_hover_tool('slice', ['name', 'center', 'duration', 'area'])

    @Graph.node(simulator=True)
    def RF_hover(self):
        return self.get_hover_tool('RF', ['name', 'center', 'duration', 'flip_angle'])

    @Graph.node(simulator=True)
    def signal_hover(self):
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
    
    @param.depends('sequence_type', 'FatSat', 'TR', 'TE', 'FA', 'TI', 'FOV_F', 'FOV_P', 'phase_oversampling', 'num_shots', 'matrix_F_param', 'matrix_P_param', 'voxel_F', 'voxel_P', 'slice_thickness', 'trajectory', 'frequency_direction', 'pixel_bandwidth_param', 'FOV_bandwidth', 'FW_shift', 'partial_Fourier', 'turbo_factor', 'EPI_factor', 'shot')
    def display_sequence_plot(self):
        return self.graph.nodes['sequence_plot'].value
    
    @param.depends('object', 'field_strength', 'sequence_type', 'FatSat', 'TR', 'TE', 'FA', 'TI', 'FOV_F', 'FOV_P', 'phase_oversampling', 'num_shots', 'matrix_F_param', 'matrix_P_param', 'voxel_F', 'voxel_P', 'recon_matrix_F_param', 'recon_matrix_P_param', 'recon_voxel_F', 'recon_voxel_P', 'slice_thickness', 'trajectory', 'frequency_direction', 'pixel_bandwidth_param', 'FOV_bandwidth', 'FW_shift', 'NSA', 'partial_Fourier', 'turbo_factor', 'EPI_factor', 'kspace_type', 'show_processed_kspace', 'kspace_exponent', 'homodyne', 'do_apodize', 'apodization_alpha', 'do_zerofill', 'radial_FOV_oversampling')
    def display_kspace(self):
        return self.graph.nodes['kspace'].value

    @param.depends('object', 'field_strength', 'sequence_type', 'FatSat', 'TR', 'TE', 'FA', 'TI', 'FOV_F', 'FOV_P', 'phase_oversampling', 'num_shots', 'matrix_F_param', 'matrix_P_param', 'voxel_F', 'voxel_P', 'recon_matrix_F_param', 'recon_matrix_P_param', 'recon_voxel_F', 'recon_voxel_P', 'slice_thickness', 'trajectory', 'frequency_direction', 'pixel_bandwidth_param', 'FOV_bandwidth', 'FW_shift', 'NSA', 'partial_Fourier', 'turbo_factor', 'EPI_factor', 'image_type', 'show_FOV', 'homodyne', 'do_apodize', 'apodization_alpha', 'do_zerofill', 'radial_FOV_oversampling')
    def display_image(self):
        return self.graph.nodes['image'].value * self.graph.nodes['FOV_box'].value