import holoviews as hv
from holoviews import streams
import param
import numpy as np
import math
from pathlib import Path
from spinsight import constants, convert, phantom
from spinsight import nodes # needed to initialize graph node decorators
from spinsight.DAG import Graph
from spinsight.params import PARAMS
from bokeh.models import HoverTool, CustomJS, ColumnDataSource
import warnings

hv.extension('bokeh')


def snap(value, values, mode='nearest'):
    match mode:
        case 'nearest':
            return min(values, key=lambda x: abs(x-value), default=None)
        case 'ceil':
            return min([v for v in values if v >= value], default=None)
        case 'floor':
            return max([v for v in values if v <= value], default=None)
        case _:
            raise ValueError(f'Invalid mode {mode}')


def filter_objects(objects, minval=None, maxval=None):
    if isinstance(objects, dict):
        return {k: v for k, v in objects.items() if (minval or -np.inf) <= v <= (maxval or np.inf)}
    return [v for v in objects if (minval or -np.inf) <= v <= (maxval or np.inf)]


def value_in_objects(value, objects):
    if isinstance(objects, dict):
        return value in objects.values()
    return value in objects


def insert_value_in_list_sorted(value, list_):
    list_.append(value)
    return sorted(list_)


def insert_value_in_dict_sorted(key, value, dict_):
    dict_[key] = value
    return dict(sorted(dict_.items()))


def get_object_values(objects):
    if callable(getattr(objects, 'values', False)):
        return objects.values()
    return objects


def get_k_on_interval(interval, k_trajectory):
    t = np.arange(*interval[[0, -1]], k_trajectory['dt'])
    kx = np.interp(t, k_trajectory['t'], k_trajectory['kx'])
    ky = np.interp(t, k_trajectory['t'], k_trajectory['ky'])
    return zip(kx, ky)


class MRIsimulator(param.Parameterized):

    # Settings
    object = param.ObjectSelector(**PARAMS['object'].param_kwargs)
    field_strength = param.ObjectSelector(**PARAMS['field_strength'].param_kwargs)
    parameter_style = param.ObjectSelector(**PARAMS['parameter_style'].param_kwargs)

    min_voxel_size = param.Number(**PARAMS['min_voxel_size'].param_kwargs)
    noise_gain = param.Number(**PARAMS['noise_gain'].param_kwargs)

    # Sequence
    sequence_type = param.ObjectSelector(**PARAMS['sequence_type'].param_kwargs)
    pixel_bandwidth_ui = param.Selector(**PARAMS['pixel_bandwidth_ui'].param_kwargs)
    FOV_bandwidth = param.Selector(**PARAMS['FOV_bandwidth'].param_kwargs)
    FW_shift = param.Selector(**PARAMS['FW_shift'].param_kwargs)
    NSA = param.Integer(**PARAMS['NSA'].param_kwargs)
    partial_Fourier = param.Number(**PARAMS['partial_Fourier'].param_kwargs)
    turbo_factor = param.Integer(**PARAMS['turbo_factor'].param_kwargs)
    EPI_factor = param.Selector(**PARAMS['EPI_factor'].param_kwargs)

    # Contrast
    FatSat = param.Boolean(**PARAMS['FatSat'].param_kwargs)
    TR = param.Selector(**PARAMS['TR'].param_kwargs)
    TE = param.Selector(**PARAMS['TE'].param_kwargs)
    TI = param.Selector(**PARAMS['TI'].param_kwargs)
    FA = param.Selector(**PARAMS['FA'].param_kwargs)
    
    # Geometry
    trajectory = param.ObjectSelector(**PARAMS['trajectory'].param_kwargs)
    frequency_direction = param.ObjectSelector(**PARAMS['frequency_direction'].param_kwargs)
    FOV_P = param.Selector(**PARAMS['FOV_P'].param_kwargs)
    FOV_F = param.Selector(**PARAMS['FOV_F'].param_kwargs)
    phase_oversampling = param.Selector(**PARAMS['phase_oversampling'].param_kwargs)
    radial_factor = param.Number(**PARAMS['radial_factor'].param_kwargs)
    num_shots = param.Integer(**PARAMS['num_shots'].param_kwargs)
    matrix_P_ui = param.Selector(**PARAMS['matrix_P_ui'].param_kwargs)
    matrix_F_ui = param.Selector(**PARAMS['matrix_F_ui'].param_kwargs)
    voxel_P = param.Selector(**PARAMS['voxel_P'].param_kwargs)
    voxel_F = param.Selector(**PARAMS['voxel_F'].param_kwargs)
    recon_matrix_P_ui = param.Selector(**PARAMS['recon_matrix_P_ui'].param_kwargs)
    recon_matrix_F_ui = param.Selector(**PARAMS['recon_matrix_F_ui'].param_kwargs)
    recon_voxel_P = param.Selector(**PARAMS['recon_voxel_P'].param_kwargs)
    recon_voxel_F = param.Selector(**PARAMS['recon_voxel_F'].param_kwargs)
    slice_thickness = param.Selector(**PARAMS['slice_thickness'].param_kwargs)
    
    shot_label = param.String(**PARAMS['shot_label'].param_kwargs)
    radial_FOV_oversampling = param.Number(**PARAMS['radial_FOV_oversampling'].param_kwargs)
    rec_acq_ratio_P = param.Number(**PARAMS['rec_acq_ratio_P'].param_kwargs)
    rec_acq_ratio_F = param.Number(**PARAMS['rec_acq_ratio_F'].param_kwargs)
    
    # MR image
    show_FOV = param.Boolean(**PARAMS['show_FOV'].param_kwargs)
    reference_tissue = param.ObjectSelector(**PARAMS['reference_tissue'].param_kwargs)

    image_type = param.ObjectSelector(**PARAMS['image_type'].param_kwargs)
    SNR = param.Number(**PARAMS['SNR'].param_kwargs)
    reference_SNR = param.Number(**PARAMS['reference_SNR'].param_kwargs)
    relative_SNR = param.Number(**PARAMS['relative_SNR'].param_kwargs)
    scantime = param.String(**PARAMS['scantime'].param_kwargs)

    # k-space
    show_processed_kspace = param.Boolean(**PARAMS['show_processed_kspace'].param_kwargs)
    kspace_exponent = param.Number(**PARAMS['kspace_exponent'].param_kwargs)
    kspace_type = param.ObjectSelector(**PARAMS['kspace_type'].param_kwargs)

    # Post-processing
    homodyne = param.Boolean(**PARAMS['homodyne'].param_kwargs)
    do_apodize = param.Boolean(**PARAMS['do_apodize'].param_kwargs)
    apodization_alpha = param.Number(**PARAMS['apodization_alpha'].param_kwargs)
    do_zerofill = param.Boolean(**PARAMS['do_zerofill'].param_kwargs)
    
    # Sequence plot
    shot = param.Integer(**PARAMS['shot'].param_kwargs)
    spoke_angle = param.Number(**PARAMS['spoke_angle'].param_kwargs)

    def __init__(self, **params):
        
        super().__init__(**params)

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

        self.graph = Graph(self)

        self.set_reference_SNR()

    def get_params(self):
        return {param: self.__getattribute__(param) for param in self.param.values().keys() if param != 'name' and not PARAMS[param].derived}

    def set_params(self, settings):
        self.init_bounds()
        self.param.update(settings)
    
    def set_param_discrete_bounds(self, par, curval, minval=None, maxval=None):
        objects = filter_objects(PARAMS[par.name].objects, minval, maxval)
        if not value_in_objects(curval, objects):
            warnings.warn(f'Trying to set {par.name} bound [{minval}, {maxval}] outside current value: {curval})')
            if isinstance(objects, dict):
                cur_label = next((k for k, v in par.names.items() if v==curval), str(curval))
                insert_value_in_dict_sorted(cur_label, curval, objects)
            else:
                insert_value_in_list_sorted(curval, objects)
        par.objects = objects

    def set_param_bounds(self, par, minval=None, maxval=None):
        curval = getattr(self, par.name)
        if PARAMS[par.name].objects is not None:
            return self.set_param_discrete_bounds(par, curval, minval, maxval)
        
        outbound = False
        if curval < minval:
            warnings.warn(f'trying to set {par.name} min bound above current value ({minval} > {curval})')
            outbound = True
        if curval > maxval:
            warnings.warn(f'trying to set {par.name} max bound below current value ({maxval} < {curval})')
            outbound = True
        if not outbound:
            par.bounds = (minval, maxval)

    def set_visibility(self, par_name, visible):
        precedence = abs(self.param[par_name].precedence)
        if not visible:
            precedence *= -1
        self.param[par_name].precedence = precedence
    
    def set_param(self, par, value, mode='nearest'):
        objects = getattr(par, 'objects', None) # par.objects could be dict or param.ListProxy
        values = get_object_values(objects)
        new = snap(value, values, mode) if values else value

        insert_value = False
        if values and new is None:
            default_objects = PARAMS[par.name].objects 
            default_values = get_object_values(default_objects)
            new = snap(value, default_values, mode)
            if new is None:
                raise ValueError(f'Value {value} is not supported by current or default objects for param {par.name} (mode={mode})')
            insert_value = True
            if isinstance(objects, dict):
                new_label = next((k for k, v in default_objects.items() if v==new), str(new))
                insert_value_in_dict_sorted(new_label, new, objects)
            else:
                insert_value_in_list_sorted(new, objects)
        
        if new != getattr(self, par.name):
            if insert_value:
                par.objects = objects
            setattr(self, par.name, new)
    
    @Graph.node(action_precedence=0.5, simulator_method=True)
    def resolve_conflicts(self, min_TE, TE, min_TR, TR):
        if min_TE > TE:
            self.set_param(self.param.TE, min_TE, mode='ceil')
        elif min_TR > TR:
            self.set_param(self.param.TR, min_TR, mode='ceil')
    
    @Graph.node(action_precedence=1, simulator_method=True)
    def set_isotropic_voxel_size(self, is_radial, FOV_F, matrix_F, FOV_P, matrix_P):
        # TODO: needs repair, for instance set param according to param_style
        if is_radial:
            if (FOV_F / matrix_F < FOV_P / matrix_P):
                self.set_param(self.param.matrix_P, matrix_F * FOV_P / FOV_F, mode='nearest')
            else:
                self.set_param(self.param.matrix_F, matrix_P * FOV_F / FOV_P, mode='nearest')

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_x_y_labels(self, frequency_direction):
        for p in [self.param.FOV_F, self.param.FOV_P, self.param.matrix_F_ui, self.param.matrix_P_ui, self.param.recon_matrix_F_ui, self.param.recon_matrix_P_ui]:
            if (' y' in p.label) and (('_F' in p.name and frequency_direction=='left-right') or
                                        ('_P' in p.name and frequency_direction=='anterior-posterior')):
                p.label = p.label.replace(' y', ' x')
            elif (' x' in p.label) and (('_P' in p.name and frequency_direction=='left-right') or
                                        ('_F' in p.name and frequency_direction=='anterior-posterior')):
                p.label = p.label.replace(' x', ' y')

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_labels_by_trajectory(self, shot_label):
        self.param.shot.label = f'Displayed {shot_label}'
        self.param.radial_factor.label = f'{shot_label.capitalize()} sampling factor'

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_shot_label(self, shot_label):
        self.set_param(self.param.shot_label, shot_label)
    
    @Graph.node(action_precedence=1, simulator_method=True)
    def set_spoke_angle(self, spoke_angle):
        self.set_param(self.param.spoke_angle, spoke_angle)
    
    @Graph.node(action_precedence=1, simulator_method=True)
    def set_num_shots(self, num_shots):
        self.set_param(self.param.num_shots, num_shots)
    
    @Graph.node(action_precedence=1, simulator_method=True)
    def set_relative_SNR(self, relative_SNR):
        self.set_param(self.param.relative_SNR, relative_SNR)
    
    @Graph.node(action_precedence=1, simulator_method=True)
    def set_scantime(self, scantime):
        self.set_param(self.param.scantime, scantime)
    
    @Graph.node(action_precedence=1, simulator_method=True)
    def set_pixel_bandwidth(self, pixel_BW_is_input, pixel_bandwidth):
        if not pixel_BW_is_input:
            self.set_param(self.param.pixel_bandwidth_ui, pixel_bandwidth)
    
    @Graph.node(action_precedence=1, simulator_method=True)
    def set_FOV_bandwidth(self, FOV_BW_is_input, pixel_bandwidth, matrix_F):
        if not FOV_BW_is_input:
            self.set_param(self.param.FOV_bandwidth, convert.pixel_BW_to_FOV_BW(pixel_bandwidth, matrix_F))
    
    @Graph.node(action_precedence=1, simulator_method=True)
    def set_FW_shift(self, FW_shift_is_input, pixel_bandwidth, field_strength):
        if not FW_shift_is_input:
            self.set_param(self.param.FW_shift, convert.pixel_BW_to_shift(pixel_bandwidth, field_strength))
    
    @Graph.node(action_precedence=1, simulator_method=True)
    def set_matrix_F(self, matrix_is_input, matrix_F):
        if not matrix_is_input:
            self.set_param(self.param.matrix_F_ui, matrix_F)

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_matrix_P(self, matrix_is_input, matrix_P):
        if not matrix_is_input:
            self.set_param(self.param.matrix_P_ui, matrix_P)

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_recon_matrix_F(self, matrix_is_input, keep_rec_acq_ratio, recon_matrix_F):
        if not matrix_is_input or keep_rec_acq_ratio:
            self.set_param(self.param.recon_matrix_F_ui, recon_matrix_F)

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_recon_matrix_P(self, matrix_is_input, keep_rec_acq_ratio, recon_matrix_P):
        if not matrix_is_input or keep_rec_acq_ratio:
            self.set_param(self.param.recon_matrix_P_ui, recon_matrix_P)

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_voxel_F(self, voxel_size_is_input, FOV_F, matrix_F):
        if not voxel_size_is_input:
            self.set_param(self.param.voxel_F, FOV_F / matrix_F)

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_voxel_P(self, voxel_size_is_input, FOV_P, matrix_P):
        if not voxel_size_is_input:
            self.set_param(self.param.voxel_P, FOV_P / matrix_P)

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_recon_voxel_F(self, voxel_size_is_input, keep_rec_acq_ratio, FOV_F, recon_matrix_F):
        if not voxel_size_is_input or keep_rec_acq_ratio:
            self.set_param(self.param.recon_voxel_F, FOV_F / recon_matrix_F)

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_recon_voxel_P(self, voxel_size_is_input, keep_rec_acq_ratio, FOV_P, recon_matrix_P):
        if not voxel_size_is_input or keep_rec_acq_ratio:
            self.set_param(self.param.recon_voxel_P, FOV_P / recon_matrix_P)
    
    @Graph.node(action_precedence=1, simulator_method=True)
    def set_rec_acq_ratio_F(self, recon_matrix_F, matrix_F):
        self.set_param(self.param.rec_acq_ratio_F, recon_matrix_F / matrix_F)

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_rec_acq_ratio_P(self, recon_matrix_P, matrix_P):
        self.set_param(self.param.rec_acq_ratio_P, recon_matrix_P / matrix_P)
    
    @Graph.node(action_precedence=1, simulator_method=True)
    def set_shot(self, num_shots):
        if self.shot > num_shots:
            self.shot = min(self.shot, num_shots)

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_TR_bounds(self, min_TR):
        self.set_param_bounds(self.param.TR, minval=min_TR)
        
    @Graph.node(action_precedence=1, simulator_method=True)
    def set_TE_bounds(self, min_TE, max_TE):
        self.set_param_bounds(self.param.TE, minval=min_TE, maxval=max_TE)

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_TI_bounds(self, sequence_type, max_TI):
        if sequence_type == 'Inversion Recovery':
            self.set_param_bounds(self.param.TI, maxval=max_TI)

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_slice_thickness_bounds(self, min_slice_thickness):
        self.set_param_bounds(self.param.slice_thickness, minval=min_slice_thickness)

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_pixel_bandwidth_bounds(self, pixel_BW_is_input, pixel_bandwidth_bounds):
        if pixel_BW_is_input:
            self.set_param_bounds(self.param.pixel_bandwidth_ui, minval=pixel_bandwidth_bounds.min, maxval=pixel_bandwidth_bounds.max)

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_FOV_bandwidth_bounds(self, FOV_BW_is_input, pixel_bandwidth_bounds, matrix_F):
        if FOV_BW_is_input:
            self.set_param_bounds(self.param.FOV_bandwidth, minval=convert.pixel_BW_to_FOV_BW(pixel_bandwidth_bounds.min, matrix_F), maxval=convert.pixel_BW_to_FOV_BW(pixel_bandwidth_bounds.max, matrix_F))
    
    @Graph.node(action_precedence=1, simulator_method=True)
    def set_FW_shift_bounds(self, FW_shift_is_input, pixel_bandwidth_bounds, field_strength):
        if FW_shift_is_input:
            self.set_param_bounds(self.param.FW_shift, minval=convert.pixel_BW_to_shift(pixel_bandwidth_bounds.max, field_strength), maxval=convert.pixel_BW_to_shift(pixel_bandwidth_bounds.min, field_strength))

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_matrix_F_bounds(self, matrix_is_input, matrix_F_bounds):
        if matrix_is_input:
            self.set_param_bounds(self.param.matrix_F_ui, minval=matrix_F_bounds.min, maxval=matrix_F_bounds.max)

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_matrix_P_bounds(self, matrix_is_input, matrix_P_bounds):
        if matrix_is_input:
            self.set_param_bounds(self.param.matrix_P_ui, minval=matrix_P_bounds.min, maxval=matrix_P_bounds.max)

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_recon_matrix_F_bounds(self, matrix_is_input, recon_matrix_F_bounds):
        if matrix_is_input:
            self.set_param_bounds(self.param.recon_matrix_F_ui, minval=recon_matrix_F_bounds.min, maxval=recon_matrix_F_bounds.max)

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_recon_matrix_P_bounds(self, matrix_is_input, recon_matrix_P_bounds):
        if matrix_is_input:
            self.set_param_bounds(self.param.recon_matrix_P_ui, minval=recon_matrix_P_bounds.min, maxval=recon_matrix_P_bounds.max)

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_FOV_F_bounds(self, FOV_F_bounds):
        self.set_param_bounds(self.param.FOV_F, minval=FOV_F_bounds.min, maxval=FOV_F_bounds.max)

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_FOV_P_bounds(self, FOV_P_bounds):
        self.set_param_bounds(self.param.FOV_P, minval=FOV_P_bounds.min, maxval=FOV_P_bounds.max)

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_voxel_F_bounds(self, voxel_size_is_input, FOV_F, matrix_F_bounds):
        if voxel_size_is_input:
            self.set_param_bounds(self.param.voxel_F, minval=FOV_F/matrix_F_bounds.max, maxval=FOV_F/matrix_F_bounds.min)

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_voxel_P_bounds(self, voxel_size_is_input, FOV_P, matrix_P_bounds):
        if voxel_size_is_input:
            self.set_param_bounds(self.param.voxel_P, minval=FOV_P/matrix_P_bounds.max, maxval=FOV_P/matrix_P_bounds.min)
    
    @Graph.node(action_precedence=1, simulator_method=True)
    def set_recon_voxel_F_bounds(self, voxel_size_is_input, FOV_F, recon_matrix_F_bounds):
        if voxel_size_is_input:
            self.set_param_bounds(self.param.recon_voxel_F, minval=FOV_F/recon_matrix_F_bounds.max, maxval=FOV_F/recon_matrix_F_bounds.min)

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_recon_voxel_P_bounds(self, voxel_size_is_input, FOV_P, recon_matrix_P_bounds):
        if voxel_size_is_input:
            self.set_param_bounds(self.param.recon_voxel_P, minval=FOV_P/recon_matrix_P_bounds.max, maxval=FOV_P/recon_matrix_P_bounds.min)

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_turbo_factor_bounds(self, max_turbo_factor):
        # turbo_factor must equal 1 when the EPI_factor is even
        if not self.EPI_factor%2:
            self.param.turbo_factor.bounds = (1, 1)
            self.param.turbo_factor.constant = True
            return
        self.param.turbo_factor.bounds = (1, min(max_turbo_factor, PARAMS['turbo_factor'].bounds[-1]))
        self.param.turbo_factor.constant = False

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_shot_bounds(self, num_shots):
        self.param.shot.bounds = (1, num_shots)
    
    @Graph.node(action_precedence=1, simulator_method=True)
    def set_EPI_factor_objects(self, max_EPI_factor):
        self.set_param_bounds(self.param.EPI_factor, maxval=max_EPI_factor)
        # EPI_factor must be odd for turbo spin echo (GRASE)
        if self.turbo_factor > 1:
            self.param.EPI_factor.objects = [v for v in self.param.EPI_factor.objects if v%2]

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_reference_tissue_objects(self, tissues):
        self.param.reference_tissue.objects = tissues
        self.reference_tissue = tissues[0]

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_trajectory_objects(self, EPI_factor, turbo_factor):
        # Label radial trajectory 'Radial' or 'PROPELLER' depending on nLines per shot
        self.param.trajectory.objects = PARAMS['trajectory'].objects
        invalid, updated = ('PROPELLER', 'Radial') if (EPI_factor * turbo_factor == 1) else ('Radial', 'PROPELLER')
        if self.trajectory == invalid:
            self.trajectory = updated
        self.param.trajectory.objects = [t for t in PARAMS['trajectory'].objects if t != invalid]

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_pixel_bandwidth_visibility(self, pixel_BW_is_input):
        self.set_visibility('pixel_bandwidth_ui', pixel_BW_is_input)
    
    @Graph.node(action_precedence=1, simulator_method=True)
    def set_FOV_bandwidth_visibility(self, FOV_BW_is_input):
        self.set_visibility('FOV_bandwidth', FOV_BW_is_input)

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_FW_shift_visibility(self, FW_shift_is_input):
        self.set_visibility('FW_shift', FW_shift_is_input)

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_voxel_size_visibility(self, voxel_size_is_input):
        for voxel_size_param in ['voxel_F', 'voxel_P', 'recon_voxel_F', 'recon_voxel_P']:
            self.set_visibility(voxel_size_param, voxel_size_is_input)

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_matrix_visibility(self, matrix_is_input):
        for matrix_param in ['matrix_F_ui', 'matrix_P_ui', 'recon_matrix_F_ui', 'recon_matrix_P_ui']:
            self.set_visibility(matrix_param, matrix_is_input)

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_partial_Fourier_visibility(self, is_radial):
        visible = not is_radial
        self.set_visibility('partial_Fourier', visible)
        if not visible:
            self.set_param(self.param.partial_Fourier, 1)

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_frequency_direction_visibility(self, is_radial):
        self.set_visibility('frequency_direction', not is_radial)

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_phase_oversampling_visibility(self, is_radial):
        visible = not is_radial
        self.set_visibility('phase_oversampling', visible)
        if not visible:
            self.set_param(self.param.phase_oversampling, 1)

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_radial_factor_visibility(self, is_radial):
        self.set_visibility('radial_factor', is_radial)

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_TI_visibility(self, sequence_type):
        self.set_visibility('TI', sequence_type == 'Inversion Recovery')

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_FA_visibility(self, sequence_type):
        self.set_visibility('FA', sequence_type == 'Spoiled Gradient Echo')

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_turbo_factor_visibility(self, sequence_type):
        visible = sequence_type != 'Spoiled Gradient Echo'
        self.set_visibility('turbo_factor', visible)
        if not visible:
            self.set_param(self.param.turbo_factor, 1)

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_homodyne_visibility(self, num_blank_lines, is_radial):
        self.set_visibility('homodyne', (num_blank_lines > 0 and not is_radial))

    @Graph.node(action_precedence=1, simulator_method=True)
    def set_apodization_alpha_visibility(self, do_apodize):
        self.set_visibility('apodization_alpha', do_apodize)

    def get_hover_tool(self, board, attributes):
        with open(Path(__file__).parent / 'hoverCallback.js', 'r') as file:
            hover_callback = CustomJS(args={'hover_index': self.hover_index, 'board': board}, code=file.read())
        if board == 'slice':
            hover_callback = None
        return HoverTool(tooltips=[(attr, f'@{attr}') for attr in attributes], attachment='below', callback=hover_callback)

    @Graph.node(simulator_method=True)
    def frequency_hover(self):
        return self.get_hover_tool('frequency', ['name', 'center', 'duration', 'area'])

    @Graph.node(simulator_method=True)
    def phase_hover(self):
        return self.get_hover_tool('phase', ['name', 'center', 'duration', 'area'])

    @Graph.node(simulator_method=True)
    def slice_hover(self):
        return self.get_hover_tool('slice', ['name', 'center', 'duration', 'area'])

    @Graph.node(simulator_method=True)
    def RF_hover(self):
        return self.get_hover_tool('RF', ['name', 'center', 'duration', 'flip_angle'])

    @Graph.node(simulator_method=True)
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
    
    @param.depends('sequence_type', 'FatSat', 'TR', 'TE', 'FA', 'TI', 'FOV_F', 'FOV_P', 'phase_oversampling', 'num_shots', 'matrix_F_ui', 'matrix_P_ui', 'voxel_F', 'voxel_P', 'slice_thickness', 'trajectory', 'frequency_direction', 'pixel_bandwidth_ui', 'FOV_bandwidth', 'FW_shift', 'partial_Fourier', 'turbo_factor', 'EPI_factor', 'shot')
    def display_sequence_plot(self):
        return self.graph.nodes['sequence_plot'].value
    
    @param.depends('object', 'field_strength', 'sequence_type', 'FatSat', 'TR', 'TE', 'FA', 'TI', 'FOV_F', 'FOV_P', 'phase_oversampling', 'num_shots', 'matrix_F_ui', 'matrix_P_ui', 'voxel_F', 'voxel_P', 'recon_matrix_F_ui', 'recon_matrix_P_ui', 'recon_voxel_F', 'recon_voxel_P', 'slice_thickness', 'trajectory', 'frequency_direction', 'pixel_bandwidth_ui', 'FOV_bandwidth', 'FW_shift', 'NSA', 'partial_Fourier', 'turbo_factor', 'EPI_factor', 'kspace_type', 'show_processed_kspace', 'kspace_exponent', 'homodyne', 'do_apodize', 'apodization_alpha', 'do_zerofill', 'radial_FOV_oversampling')
    def display_kspace(self):
        return self.graph.nodes['kspace'].value

    @param.depends('object', 'field_strength', 'sequence_type', 'FatSat', 'TR', 'TE', 'FA', 'TI', 'FOV_F', 'FOV_P', 'phase_oversampling', 'num_shots', 'matrix_F_ui', 'matrix_P_ui', 'voxel_F', 'voxel_P', 'recon_matrix_F_ui', 'recon_matrix_P_ui', 'recon_voxel_F', 'recon_voxel_P', 'slice_thickness', 'trajectory', 'frequency_direction', 'pixel_bandwidth_ui', 'FOV_bandwidth', 'FW_shift', 'NSA', 'partial_Fourier', 'turbo_factor', 'EPI_factor', 'image_type', 'show_FOV', 'homodyne', 'do_apodize', 'apodization_alpha', 'do_zerofill', 'radial_FOV_oversampling')
    def display_image(self):
        return self.graph.nodes['image'].value * self.graph.nodes['FOV_box'].value