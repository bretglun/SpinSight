import holoviews as hv
from holoviews import streams
import param
import numpy as np
import math
from pathlib import Path
from functools import partial
from spinsight import params
from spinsight.params import PARAMS
from spinsight.InputParams import InputParams
from spinsight.DAG import Graph
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
    shot_label = param.String() # shot/spoke/blade label
    num_shots = param.Integer() # number of shots
    spoke_angle = param.Number() # spoke angle [°]
    rec_acq_ratio_P = param.Number(default=2.0) # reconstructed / acquired matrix_P ratio
    rec_acq_ratio_F = param.Number(default=2.0) # reconstructed / acquired matrix_F ratio
    
    SNR = param.Number()
    reference_SNR = param.Number()
    relative_SNR = param.Number() # [%]
    scantime = param.String()

    # Dynamic map triggers
    image_update = param.Integer(default = 0)
    kspace_update = param.Integer(default = 0)
    seqplot_update = param.Integer(default = 0)

    def __init__(self, **params):
        
        super().__init__(**params)

        self.input = InputParams()

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
        self.add_input_watchers(self.graph)

        self.set_reference_SNR()

    def input_nodes(self):
        input_nodes = set(par for par in self.input.param if par != 'name')
        input_nodes.update(('rec_acq_ratio_P', 'rec_acq_ratio_F', 'reference_SNR'))
        return input_nodes
    
    def get_input_node_specs(self):
        specs = {}
        for par in self.input_nodes():
            if par in self.param:
                specs[par] = {'func': partial(getattr, self, par)}
            elif par in self.input.param:
                specs[par] = {'func': partial(getattr, self.input, par)}
        return specs
    
    def add_input_watchers(self, graph):
        for par in self.input_nodes():
            node = graph.nodes[par]
            if par in self.param:
                self.param.watch(partial(graph.on_change, node), node.name)
            elif par in self.input.param:
                self.input.param.watch(partial(graph.on_change, node), node.name)

    def get_input_params(self):
        return {par: getattr(self.input, par) for par in self.input.param if par != 'name' and not PARAMS[par].derived}

    def set_input_params(self, settings):
        self.input.param.update(settings)

    def set_visibility(self, par_name, visible):
        par = self.input.param[par_name]
        precedence = abs(par.precedence)
        if not visible:
            precedence *= -1
        par.precedence = precedence
    
    def set_param(self, par_name, value, mode='nearest'):
        par = self.input.param[par_name]
        objects = getattr(par, 'objects', None) # par.objects could be dict or param.ListProxy
        values = get_object_values(objects)
        new = params.snap(value, values, mode) if values else value

        insert_value = False
        if values and new is None:
            default_objects = PARAMS[par_name].objects 
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
        
        if new != getattr(self.input, par_name):
            if insert_value:
                par.objects = objects
            setattr(self.input, par_name, new)
    
    def set_param_bounds(self, par_name, minval=None, maxval=None):
        par = self.input.param[par_name]
        curval = getattr(self.input, par_name)
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