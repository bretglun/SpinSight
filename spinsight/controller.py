import param
from spinsight.nodes.helpers import FOV_BW_is_input, FW_shift_is_input, matrix_is_input, voxel_size_is_input
from spinsight.nodes.input import parameter_style
from spinsight.nodes.internal_input_params import TR, isotropic_voxel_size, keep_rec_acq_ratio
from spinsight.nodes.param_bounds import max_TE
from spinsight.param_utils import snap, filter_objects, value_in_objects, insert_value_in_list_sorted, insert_value_in_dict_sorted, get_object_values
from spinsight.params import PARAMS
from spinsight.input_params import InputParams
from spinsight import simulator, convert, formatting
import warnings


class Controller(param.Parameterized):
    shot_label = param.String() # shot/spoke/blade label
    rec_acq_ratio_P = param.Number(default=2.0) # reconstructed / acquired matrix_P ratio
    rec_acq_ratio_F = param.Number(default=2.0) # reconstructed / acquired matrix_F ratio
    
    reference_SNR = param.Number()

    def __init__(self, gui, **params):
        super().__init__(**params)
        self.gui = gui
        self.input = InputParams()
        self.graph = simulator.make_graph(self, gui)
        self.add_input_watchers()
        self.set_reference_SNR()
    
    def set_reference_SNR(self, event=None):
        self.reference_SNR = self.from_graph('SNR')
        
    def input_nodes(self):
        return {p for p in (set(self.input.param) | set(self.param)) if p in self.graph.nodes and not self.graph.nodes[p].parents}
    
    def add_input_watchers(self):
        for par in self.input_nodes():
            if par in self.param:
                self.param.watch(self.on_param_change, par)
            elif par in self.input.param:
                self.input.param.watch(self.on_param_change, par)

    def on_param_change(self, event):
        self.graph.update_inputs({event.name: event.new})
        self.sync_with_graph()

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
        
        if minval is None:
            if PARAMS[par_name].bounds is None:
                return
            minval = PARAMS[par.name].bounds[0]
        if maxval is None:
            if PARAMS[par_name].bounds is None:
                return
            maxval = PARAMS[par.name].bounds[1]
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
        self.graph.update_inputs(settings)

    def from_graph(self, par_name):
        return self.graph.nodes[par_name].value()

    def sync_with_graph(self):
        self.update_params()
        self.update_info_params()
        self.update_plots()
        self.update_rec_acq_ratio()

    def pixel_BW_is_input(self):
        return 'PIXEL BW' in self.input.parameter_style.upper()

    def FOV_BW_is_input(self):
        return 'FOV BW' in self.input.parameter_style.upper()

    def FW_shift_is_input(self):
        return 'FAT/WATER SHIFT' in self.input.parameter_style.upper()

    def matrix_is_input(self):
        return 'MATRIX' in self.input.parameter_style.upper()

    def voxel_size_is_input(self):
        return 'VOXEL SIZE' in self.input.parameter_style.upper()

    def update_params(self):
        if not self.pixel_BW_is_input():
            self.set_param('pixel_bandwidth_ui', self.from_graph('pixel_bandwidth'))
        if not self.FOV_BW_is_input():
            self.set_param('FOV_bandwidth', convert.pixel_BW_to_FOV_BW(self.from_graph('pixel_bandwidth'), self.from_graph('matrix_F')))
        if not self.FW_shift_is_input():
            self.set_param('FW_shift_ui', self.from_graph('FW_shift'))
        if not self.matrix_is_input() or self.from_graph('isotropic_voxel_size'):
            self.set_param('matrix_F_ui', self.from_graph('matrix_F'))
            self.set_param('matrix_P_ui', self.from_graph('matrix_P'))
        if 'object' in self.from_graph('trigger_nodes'):
            if self.from_graph('FOV_F') < self.from_graph('phantom_object')['support'][self.from_graph('freq_dir')]:
                self.set_param('FOV_F', self.from_graph('phantom_object')['support'][self.from_graph('freq_dir')], mode='ceil')
            if self.from_graph('FOV_P') < self.from_graph('phantom_object')['support'][self.from_graph('phase_dir')]:
                self.set_param('FOV_P', self.from_graph('phantom_object')['support'][self.from_graph('phase_dir')], mode='ceil')
        if not self.matrix_is_input() or self.from_graph('keep_rec_acq_ratio'):
            self.set_param('recon_matrix_F_ui', self.from_graph('recon_matrix_F'))
            self.set_param('recon_matrix_P_ui', self.from_graph('recon_matrix_P'))
        if not self.voxel_size_is_input() or self.from_graph('isotropic_voxel_size'):
            self.set_param('voxel_F', self.from_graph('FOV_F') / self.from_graph('matrix_F'))
            self.set_param('voxel_P', self.from_graph('FOV_P') / self.from_graph('matrix_P'))
        if not self.voxel_size_is_input() or self.from_graph('keep_rec_acq_ratio'):
            self.set_param('recon_voxel_F', self.from_graph('FOV_F') / self.from_graph('recon_matrix_F'))
            self.set_param('recon_voxel_P', self.from_graph('FOV_P') / self.from_graph('recon_matrix_P'))
        self.set_param_bounds('TR_ui', minval=self.from_graph('min_TR'))
        self.set_param('TR_ui', self.from_graph('TR'))
        self.set_param_bounds('TE_ui', minval=self.from_graph('min_TE'), maxval=self.from_graph('max_TE'))
        self.set_param('TE_ui', self.from_graph('TE'))
        self.input.param.shot_ui.bounds = (1, self.from_graph('num_shots'))
        self.set_param('shot_ui', self.from_graph('shot') + 1)

        # Label radial trajectory 'Radial' or 'PROPELLER' depending on nLines per shot
        invalid, updated = ('PROPELLER', 'Radial') if (self.from_graph('EPI_factor') * self.from_graph('turbo_factor') == 1) else ('Radial', 'PROPELLER')
        if self.input.trajectory == invalid:
            self.input.param.trajectory.objects = PARAMS['trajectory'].objects
            self.input.trajectory = updated
        self.input.param.trajectory.objects = [t for t in PARAMS['trajectory'].objects if t != invalid]

    def update_info_params(self):
        for par in ['spoke_angle', 'num_shots', 'relative_SNR', 'scantime', 'pixel_bandwidth', 'FW_shift']:
            setattr(self.gui, par, getattr(formatting, par)(self.from_graph(par)))

    def update_plots(self):
        self.gui.image = self.from_graph('annotated_image')
        self.gui.kspace = self.from_graph('kspace')
        self.gui.hover.k_trajectory = self.from_graph('k_trajectory')
        for board in ['frequency', 'phase', 'RF', 'signal']:
            self.gui.hover.objects[board] = self.from_graph(f'{board}_objects')
        self.gui.sequence_plot = self.from_graph('sequence_plot')

    def update_rec_acq_ratio(self):
        self.rec_acq_ratio_F = self.from_graph('recon_matrix_F') / self.from_graph('matrix_F')
        self.rec_acq_ratio_P = self.from_graph('recon_matrix_P') / self.from_graph('matrix_P')