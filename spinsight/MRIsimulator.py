import holoviews as hv
from holoviews import streams
import param
import numpy as np
import math
from pathlib import Path
from spinsight import params
from spinsight.DAG import Graph
from spinsight.params import PARAMS
from bokeh.models import HoverTool, CustomJS, ColumnDataSource
import warnings
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
    SNR_and_scantime,
    set_ui_param_visibility,
    set_ui_param_bounds,
    set_ui_params,
)


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
        
        self.graph = Graph(self)

        self.set_reference_SNR()

    def get_params(self):
        return {param: self.__getattribute__(param) for param in self.param.values().keys() if param != 'name' and not PARAMS[param].derived}

    def set_params(self, settings):
        self.init_bounds()
        self.param.update(settings)

    def set_visibility(self, par_name, visible):
        precedence = abs(self.param[par_name].precedence)
        if not visible:
            precedence *= -1
        self.param[par_name].precedence = precedence

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