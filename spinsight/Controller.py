import param
from functools import partial
from spinsight.param_utils import snap, filter_objects, value_in_objects, insert_value_in_list_sorted, insert_value_in_dict_sorted, get_object_values
from spinsight.params import PARAMS
from spinsight.InputParams import InputParams
import warnings


class Controller(param.Parameterized):
    shot_label = param.String() # shot/spoke/blade label
    num_shots = param.Integer() # number of shots
    spoke_angle = param.Number() # spoke angle [°]
    rec_acq_ratio_P = param.Number(default=2.0) # reconstructed / acquired matrix_P ratio
    rec_acq_ratio_F = param.Number(default=2.0) # reconstructed / acquired matrix_F ratio
    
    SNR = param.Number()
    reference_SNR = param.Number()
    relative_SNR = param.Number() # [%]
    scantime = param.String()

    def __init__(self, **params):
        super().__init__(**params)
        self.input = InputParams()
    
    def get_input_node_specs(self):
        specs = {}
        for par in self.input_nodes():
            if par in self.param:
                specs[par] = {'func': partial(getattr, self, par)}
            elif par in self.input.param:
                specs[par] = {'func': partial(getattr, self.input, par)}
        return specs
    
    def input_nodes(self):
        input_nodes = set(par for par in self.input.param if par != 'name')
        input_nodes.update(('rec_acq_ratio_P', 'rec_acq_ratio_F', 'reference_SNR'))
        return input_nodes
    
    def add_input_watchers(self, graph):
        for par in self.input_nodes():
            node = graph.nodes[par]
            if par in self.param:
                self.param.watch(partial(graph.on_change, node), node.name)
            elif par in self.input.param:
                self.input.param.watch(partial(graph.on_change, node), node.name)

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
        new = snap(value, values, mode) if values else value

        insert_value = False
        if values and new is None:
            default_objects = PARAMS[par_name].objects 
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
    
    def get_input_params(self):
        return {par: getattr(self.input, par) for par in self.input.param if par != 'name' and not PARAMS[par].derived}

    def set_input_params(self, settings):
        self.input.param.update(settings)