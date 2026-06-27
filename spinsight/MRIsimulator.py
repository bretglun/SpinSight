import holoviews as hv
from holoviews import streams
import param
import numpy as np
import math
from pathlib import Path
from spinsight import constants, convert, params
from spinsight.constants import ACTION
# Initialize graph node decorators:
from spinsight.nodes import (
    internal_input_params,
    helpers,
    param_bounds,
    setup_sequence_objects,
    sequence_timing,
    phase_encoding_order,
    place_sequence_objects,
    kspace_simulation,
    kspace_processing,
    image_reconstruction,
    sequence_plot,
    kspace_plot,
    image_plot,
    SNR_and_scantime
)
from spinsight.DAG import Graph
from spinsight.params import PARAMS
from bokeh.models import HoverTool, CustomJS, ColumnDataSource
import warnings

hv.extension('bokeh')


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
    TR_ui = param.Selector(**PARAMS['TR_ui'].param_kwargs)
    TE_ui = param.Selector(**PARAMS['TE_ui'].param_kwargs)
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
    shot_ui = param.Integer(**PARAMS['shot_ui'].param_kwargs)
    spoke_angle = param.Number(**PARAMS['spoke_angle'].param_kwargs)
    
    # Dynamic map triggers
    image_update = param.Integer(default = 0)
    kspace_update = param.Integer(default = 0)
    seqplot_update = param.Integer(default = 0)

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
        new = params.snap(value, values, mode) if values else value

        insert_value = False
        if values and new is None:
            default_objects = PARAMS[par.name].objects 
            default_values = get_object_values(default_objects)
            new = params.snap(value, default_values, mode)
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
        object = self.graph.nodes[f'{board}_objects'].value[index]
        k_trajectory = self.graph.nodes['k_trajectory'].value
        self.k_line.event(coords=list(get_k_on_interval(object['time'][[0, -1]], k_trajectory)))
    
    def set_reference_SNR(self, event=None):
        self.reference_SNR = self.graph.nodes['SNR'].value
    
    @param.depends('seqplot_update')
    def display_sequence_plot(self):
        return self.graph.nodes['sequence_plot'].value
    
    @param.depends('kspace_update')
    def display_kspace(self):
        return self.graph.nodes['kspace'].value
    
    @param.depends('image_update')
    def display_image(self):
        return self.graph.nodes['annotated_image'].value


@Graph.node(action=ACTION.BOUNDS)
def set_x_y_labels(simulator, frequency_direction):
    for p in [simulator.param.FOV_F, simulator.param.FOV_P, simulator.param.matrix_F_ui, simulator.param.matrix_P_ui, simulator.param.recon_matrix_F_ui, simulator.param.recon_matrix_P_ui]:
        if (' y' in p.label) and (('_F' in p.name and frequency_direction=='left-right') or
                                    ('_P' in p.name and frequency_direction=='anterior-posterior')):
            p.label = p.label.replace(' y', ' x')
        elif (' x' in p.label) and (('_P' in p.name and frequency_direction=='left-right') or
                                    ('_F' in p.name and frequency_direction=='anterior-posterior')):
            p.label = p.label.replace(' x', ' y')


@Graph.node()
def shot_label(is_radial, EPI_factor, turbo_factor):
    return 'shot' if not is_radial else 'spoke' if (EPI_factor * turbo_factor == 1) else 'blade'


@Graph.node(action=ACTION.BOUNDS)
def set_labels_by_trajectory(simulator, shot_label):
    simulator.param.shot_ui.label = f'Displayed {shot_label}'
    simulator.param.radial_factor.label = f'{shot_label.capitalize()} sampling factor'


@Graph.node(action=ACTION.BOUNDS)
def set_shot_label(simulator, shot_label):
    simulator.set_param(simulator.param.shot_label, shot_label)


@Graph.node(action=ACTION.VALUE)
def set_spoke_angle(simulator, spoke_angle):
    simulator.set_param(simulator.param.spoke_angle, spoke_angle)


@Graph.node(action=ACTION.VALUE)
def set_num_shots(simulator, num_shots):
    simulator.set_param(simulator.param.num_shots, num_shots)


@Graph.node(action=ACTION.VALUE)
def set_relative_SNR(simulator, relative_SNR):
    simulator.set_param(simulator.param.relative_SNR, relative_SNR)


@Graph.node(action=ACTION.VALUE)
def set_scantime(simulator, scantime):
    simulator.set_param(simulator.param.scantime, scantime)


@Graph.node(action=ACTION.VALUE)
def set_pixel_bandwidth(simulator, pixel_BW_is_input, pixel_bandwidth):
    if not pixel_BW_is_input:
        simulator.set_param(simulator.param.pixel_bandwidth_ui, pixel_bandwidth)


@Graph.node(action=ACTION.VALUE)
def set_FOV_bandwidth(simulator, FOV_BW_is_input, pixel_bandwidth, matrix_F):
    if not FOV_BW_is_input:
        simulator.set_param(simulator.param.FOV_bandwidth, convert.pixel_BW_to_FOV_BW(pixel_bandwidth, matrix_F))


@Graph.node(action=ACTION.VALUE)
def set_FW_shift(simulator, FW_shift_is_input, pixel_bandwidth, field_strength):
    if not FW_shift_is_input:
        simulator.set_param(simulator.param.FW_shift, convert.pixel_BW_to_shift(pixel_bandwidth, field_strength))


@Graph.node(action=ACTION.VALUE)
def set_matrix_F(simulator, matrix_is_input, isotropic_voxel_size, matrix_F):
    if not matrix_is_input or isotropic_voxel_size:
        simulator.set_param(simulator.param.matrix_F_ui, matrix_F)


@Graph.node(action=ACTION.VALUE)
def set_matrix_P(simulator, matrix_is_input, isotropic_voxel_size, matrix_P):
    if not matrix_is_input or isotropic_voxel_size:
        simulator.set_param(simulator.param.matrix_P_ui, matrix_P)


@Graph.node(action=ACTION.VALUE)
def set_recon_matrix_F(simulator, matrix_is_input, keep_rec_acq_ratio, recon_matrix_F):
    if not matrix_is_input or keep_rec_acq_ratio:
        simulator.set_param(simulator.param.recon_matrix_F_ui, recon_matrix_F)


@Graph.node(action=ACTION.VALUE)
def set_recon_matrix_P(simulator, matrix_is_input, keep_rec_acq_ratio, recon_matrix_P):
    if not matrix_is_input or keep_rec_acq_ratio:
        simulator.set_param(simulator.param.recon_matrix_P_ui, recon_matrix_P)


@Graph.node(action=ACTION.VALUE)
def set_voxel_F(simulator, voxel_size_is_input, isotropic_voxel_size, FOV_F, matrix_F):
    if not voxel_size_is_input or isotropic_voxel_size:
        simulator.set_param(simulator.param.voxel_F, FOV_F / matrix_F)


@Graph.node(action=ACTION.VALUE)
def set_voxel_P(simulator, voxel_size_is_input, isotropic_voxel_size, FOV_P, matrix_P):
    if not voxel_size_is_input or isotropic_voxel_size:
        simulator.set_param(simulator.param.voxel_P, FOV_P / matrix_P)


@Graph.node(action=ACTION.VALUE)
def set_recon_voxel_F(simulator, voxel_size_is_input, keep_rec_acq_ratio, FOV_F, recon_matrix_F):
    if not voxel_size_is_input or keep_rec_acq_ratio:
        simulator.set_param(simulator.param.recon_voxel_F, FOV_F / recon_matrix_F)


@Graph.node(action=ACTION.VALUE)
def set_recon_voxel_P(simulator, voxel_size_is_input, keep_rec_acq_ratio, FOV_P, recon_matrix_P):
    if not voxel_size_is_input or keep_rec_acq_ratio:
        simulator.set_param(simulator.param.recon_voxel_P, FOV_P / recon_matrix_P)


@Graph.node(action=ACTION.INVISIBLE)
def set_rec_acq_ratio_F(simulator, recon_matrix_F, matrix_F):
    simulator.set_param(simulator.param.rec_acq_ratio_F, recon_matrix_F / matrix_F)


@Graph.node(action=ACTION.INVISIBLE)
def set_rec_acq_ratio_P(simulator, recon_matrix_P, matrix_P):
    simulator.set_param(simulator.param.rec_acq_ratio_P, recon_matrix_P / matrix_P)


@Graph.node(action=ACTION.VALUE)
def set_TR_and_bounds(simulator, min_TR, TR):
    simulator.set_param_bounds(simulator.param.TR_ui, minval=min_TR)
    simulator.set_param(simulator.param.TR_ui, TR)


@Graph.node(action=ACTION.VALUE)
def set_TE_and_bounds(simulator, min_TE, max_TE, TE):
    simulator.set_param_bounds(simulator.param.TE_ui, minval=min_TE, maxval=max_TE)
    simulator.set_param(simulator.param.TE_ui, TE)


@Graph.node(action=ACTION.BOUNDS)
def set_TI_bounds(simulator, sequence_type, max_TI):
    if sequence_type == 'Inversion Recovery':
        simulator.set_param_bounds(simulator.param.TI, maxval=max_TI)


@Graph.node(action=ACTION.BOUNDS)
def set_slice_thickness_bounds(simulator, min_slice_thickness):
    simulator.set_param_bounds(simulator.param.slice_thickness, minval=min_slice_thickness)


@Graph.node(action=ACTION.BOUNDS)
def set_pixel_bandwidth_bounds(simulator, pixel_BW_is_input, pixel_bandwidth_bounds):
    if pixel_BW_is_input:
        simulator.set_param_bounds(simulator.param.pixel_bandwidth_ui, minval=pixel_bandwidth_bounds.min, maxval=pixel_bandwidth_bounds.max)


@Graph.node(action=ACTION.BOUNDS)
def set_FOV_bandwidth_bounds(simulator, FOV_BW_is_input, pixel_bandwidth_bounds, matrix_F):
    if FOV_BW_is_input:
        simulator.set_param_bounds(simulator.param.FOV_bandwidth, minval=convert.pixel_BW_to_FOV_BW(pixel_bandwidth_bounds.min, matrix_F), maxval=convert.pixel_BW_to_FOV_BW(pixel_bandwidth_bounds.max, matrix_F))


@Graph.node(action=ACTION.BOUNDS)
def set_FW_shift_bounds(simulator, FW_shift_is_input, pixel_bandwidth_bounds, field_strength):
    if FW_shift_is_input:
        simulator.set_param_bounds(simulator.param.FW_shift, minval=convert.pixel_BW_to_shift(pixel_bandwidth_bounds.max, field_strength), maxval=convert.pixel_BW_to_shift(pixel_bandwidth_bounds.min, field_strength))


@Graph.node(action=ACTION.BOUNDS)
def set_matrix_F_bounds(simulator, matrix_is_input, matrix_F_bounds):
    if matrix_is_input:
        simulator.set_param_bounds(simulator.param.matrix_F_ui, minval=matrix_F_bounds.min, maxval=matrix_F_bounds.max)


@Graph.node(action=ACTION.BOUNDS)
def set_matrix_P_bounds(simulator, matrix_is_input, matrix_P_bounds):
    if matrix_is_input:
        simulator.set_param_bounds(simulator.param.matrix_P_ui, minval=matrix_P_bounds.min, maxval=matrix_P_bounds.max)


@Graph.node(action=ACTION.BOUNDS)
def set_recon_matrix_F_bounds(simulator, matrix_is_input, recon_matrix_F_bounds):
    if matrix_is_input:
        simulator.set_param_bounds(simulator.param.recon_matrix_F_ui, minval=recon_matrix_F_bounds.min, maxval=recon_matrix_F_bounds.max)


@Graph.node(action=ACTION.BOUNDS)
def set_recon_matrix_P_bounds(simulator, matrix_is_input, recon_matrix_P_bounds):
    if matrix_is_input:
        simulator.set_param_bounds(simulator.param.recon_matrix_P_ui, minval=recon_matrix_P_bounds.min, maxval=recon_matrix_P_bounds.max)


@Graph.node(action=ACTION.BOUNDS)
def set_FOV_F_bounds(simulator, FOV_F_bounds):
    simulator.set_param_bounds(simulator.param.FOV_F, minval=FOV_F_bounds.min, maxval=FOV_F_bounds.max)


@Graph.node(action=ACTION.BOUNDS)
def set_FOV_P_bounds(simulator, FOV_P_bounds):
    simulator.set_param_bounds(simulator.param.FOV_P, minval=FOV_P_bounds.min, maxval=FOV_P_bounds.max)


@Graph.node(action=ACTION.BOUNDS)
def set_voxel_F_bounds(simulator, voxel_size_is_input, FOV_F, matrix_F_bounds):
    if voxel_size_is_input:
        simulator.set_param_bounds(simulator.param.voxel_F, minval=FOV_F/matrix_F_bounds.max, maxval=FOV_F/matrix_F_bounds.min)


@Graph.node(action=ACTION.BOUNDS)
def set_voxel_P_bounds(simulator, voxel_size_is_input, FOV_P, matrix_P_bounds):
    if voxel_size_is_input:
        simulator.set_param_bounds(simulator.param.voxel_P, minval=FOV_P/matrix_P_bounds.max, maxval=FOV_P/matrix_P_bounds.min)


@Graph.node(action=ACTION.BOUNDS)
def set_recon_voxel_F_bounds(simulator, voxel_size_is_input, FOV_F, recon_matrix_F_bounds):
    if voxel_size_is_input:
        simulator.set_param_bounds(simulator.param.recon_voxel_F, minval=FOV_F/recon_matrix_F_bounds.max, maxval=FOV_F/recon_matrix_F_bounds.min)


@Graph.node(action=ACTION.BOUNDS)
def set_recon_voxel_P_bounds(simulator, voxel_size_is_input, FOV_P, recon_matrix_P_bounds):
    if voxel_size_is_input:
        simulator.set_param_bounds(simulator.param.recon_voxel_P, minval=FOV_P/recon_matrix_P_bounds.max, maxval=FOV_P/recon_matrix_P_bounds.min)


@Graph.node(action=ACTION.BOUNDS)
def set_turbo_factor_bounds(simulator, max_turbo_factor):
    # turbo_factor must equal 1 when the EPI_factor is even
    if not simulator.EPI_factor%2:
        simulator.param.turbo_factor.bounds = (1, 1)
        simulator.param.turbo_factor.constant = True
        return
    simulator.param.turbo_factor.bounds = (1, min(max_turbo_factor, PARAMS['turbo_factor'].bounds[-1]))
    simulator.param.turbo_factor.constant = False


@Graph.node(action=ACTION.VALUE)
def set_shot_and_bounds(simulator, num_shots, shot):
    simulator.param.shot_ui.bounds = (1, num_shots)
    simulator.set_param(simulator.param.shot_ui, shot + 1)


@Graph.node(action=ACTION.BOUNDS)
def set_EPI_factor_objects(simulator, max_EPI_factor):
    simulator.set_param_bounds(simulator.param.EPI_factor, maxval=max_EPI_factor)
    # EPI_factor must be odd for turbo spin echo (GRASE)
    if simulator.turbo_factor > 1:
        simulator.param.EPI_factor.objects = [v for v in simulator.param.EPI_factor.objects if v%2]


@Graph.node(action=ACTION.VALUE)
def set_reference_tissue_objects(simulator, tissues):
    simulator.param.reference_tissue.objects = tissues
    simulator.reference_tissue = tissues[0]


@Graph.node(action=ACTION.VALUE)
def set_trajectory_objects(simulator, EPI_factor, turbo_factor):
    # Label radial trajectory 'Radial' or 'PROPELLER' depending on nLines per shot
    simulator.param.trajectory.objects = PARAMS['trajectory'].objects
    invalid, updated = ('PROPELLER', 'Radial') if (EPI_factor * turbo_factor == 1) else ('Radial', 'PROPELLER')
    if simulator.trajectory == invalid:
        simulator.trajectory = updated
    simulator.param.trajectory.objects = [t for t in PARAMS['trajectory'].objects if t != invalid]


@Graph.node(action=ACTION.VISIBILITY)
def set_pixel_bandwidth_visibility(simulator, pixel_BW_is_input):
    simulator.set_visibility('pixel_bandwidth_ui', pixel_BW_is_input)


@Graph.node(action=ACTION.VISIBILITY)
def set_FOV_bandwidth_visibility(simulator, FOV_BW_is_input):
    simulator.set_visibility('FOV_bandwidth', FOV_BW_is_input)


@Graph.node(action=ACTION.VISIBILITY)
def set_FW_shift_visibility(simulator, FW_shift_is_input):
    simulator.set_visibility('FW_shift', FW_shift_is_input)


@Graph.node(action=ACTION.VISIBILITY)
def set_voxel_size_visibility(simulator, voxel_size_is_input):
    for voxel_size_param in ['voxel_F', 'voxel_P', 'recon_voxel_F', 'recon_voxel_P']:
        simulator.set_visibility(voxel_size_param, voxel_size_is_input)


@Graph.node(action=ACTION.VISIBILITY)
def set_matrix_visibility(simulator, matrix_is_input):
    for matrix_param in ['matrix_F_ui', 'matrix_P_ui', 'recon_matrix_F_ui', 'recon_matrix_P_ui']:
        simulator.set_visibility(matrix_param, matrix_is_input)


@Graph.node(action=ACTION.VISIBILITY)
def set_partial_Fourier_visibility(simulator, is_radial):
    visible = not is_radial
    simulator.set_visibility('partial_Fourier', visible)
    if not visible:
        simulator.set_param(simulator.param.partial_Fourier, 1)


@Graph.node(action=ACTION.VISIBILITY)
def set_frequency_direction_visibility(simulator, is_radial):
    simulator.set_visibility('frequency_direction', not is_radial)


@Graph.node(action=ACTION.VISIBILITY)
def set_phase_oversampling_visibility(simulator, is_radial):
    visible = not is_radial
    simulator.set_visibility('phase_oversampling', visible)
    if not visible:
        simulator.set_param(simulator.param.phase_oversampling, 1)


@Graph.node(action=ACTION.VISIBILITY)
def set_radial_factor_visibility(simulator, is_radial):
    simulator.set_visibility('radial_factor', is_radial)


@Graph.node(action=ACTION.VISIBILITY)
def set_TI_visibility(simulator, sequence_type):
    simulator.set_visibility('TI', sequence_type == 'Inversion Recovery')


@Graph.node(action=ACTION.VISIBILITY)
def set_FA_visibility(simulator, sequence_type):
    simulator.set_visibility('FA', sequence_type == 'Spoiled Gradient Echo')


@Graph.node(action=ACTION.VISIBILITY)
def set_turbo_factor_visibility(simulator, sequence_type):
    visible = sequence_type != 'Spoiled Gradient Echo'
    simulator.set_visibility('turbo_factor', visible)
    if not visible:
        simulator.set_param(simulator.param.turbo_factor, 1)


@Graph.node(action=ACTION.VISIBILITY)
def set_homodyne_visibility(simulator, num_blank_lines, is_radial):
    simulator.set_visibility('homodyne', (num_blank_lines > 0 and not is_radial))


@Graph.node(action=ACTION.VISIBILITY)
def set_apodization_alpha_visibility(simulator, do_apodize):
    simulator.set_visibility('apodization_alpha', do_apodize)