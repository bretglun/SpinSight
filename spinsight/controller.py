import param
from spinsight.param_utils import snap, filter_objects, value_in_objects, insert_value_in_list_sorted, insert_value_in_dict_sorted, get_object_values
from spinsight.params import PARAMS
from spinsight.input_params import InputParams
from spinsight import simulator, convert, formatting
import warnings
import numpy as np


class Controller(param.Parameterized):
    shot_label = param.String() # shot/spoke/blade label
    rec_acq_ratio_P = param.Number() # reconstructed / acquired matrix_P ratio
    rec_acq_ratio_F = param.Number() # reconstructed / acquired matrix_F ratio
    
    reference_SNR = param.Number()

    def __init__(self, gui, **params):
        super().__init__(**params)
        self.gui = gui
        self.input = InputParams()
        self.graph = simulator.make_graph()
        self.passive_updates = False # passive_updates disables watchers updating the graph
        self.add_input_watchers()
        self.set_reference_SNR()
        self.sync_with_graph()
    
    def set_reference_SNR(self, event=None):
        self.reference_SNR = self.from_graph('SNR')
        
    def input_nodes(self):
        return {par for par in self.input.param if any(par + post in self.graph.nodes and not self.graph.nodes[par + post].parents for post in ('', '_prescribed'))}
    
    def add_input_watchers(self):
        for par in self.input.param:
            self.input.param.watch(self.on_param_change, par)

    def on_param_change(self, event):
        if self.passive_updates:
            return
        self.graph.update_inputs(self.inputs_to_update(event.name, event.new))
        self.sync_with_graph()
    
    def inputs_to_update(self, triggered, value):
        inputs = {triggered: value}
        if 'parameter_style' in inputs:
            inputs['constant_voxel_bounds'] = 'VOXEL SIZE' in self.input.parameter_style.upper()
            inputs['constant_FOV_BW_bounds'] = 'FOV BW' in self.input.parameter_style.upper()
            del inputs['parameter_style']
        if 'FOV_bandwidth' in inputs:
            inputs['pixel_bandwidth'] = convert.FOV_BW_to_pixel_BW(inputs['FOV_bandwidth'], self.from_graph('matrix_F'))
            del inputs['FOV_bandwidth']
        if 'FW_shift' in inputs:
            inputs['pixel_bandwidth'] = convert.shift_to_pixel_BW(inputs['FW_shift'], self.from_graph('field_strength'))
            del inputs['FW_shift']
        for dir in ['F', 'P']:
            # replace voxel size input with matrix size
            for prefix in ['', 'recon_']:
                if f'{prefix}voxel_{dir}' in inputs:
                    inputs[f'{prefix}matrix_{dir}'] = int(np.round(self.from_graph(f'FOV_{dir}') / inputs[f'{prefix}voxel_{dir}']))
                    del inputs[f'{prefix}voxel_{dir}']
            # maintain constant rec/acq ratio when acq matrix is changed
            if f'matrix_{dir}' in inputs:
                inputs[f'recon_matrix_{dir}'] = int(np.round(inputs[f'matrix_{dir}'] * getattr(self, f'rec_acq_ratio_{dir}')))
        return inputs

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

    def sync_with_graph(self):
        self.update_visibility()
        self.update_bounds()
        self.update_params()
        self.update_info_params()
        self.update_plots()
        self.update_rec_acq_ratio()

    def update_visibility(self):
        self.set_visibility('pixel_bandwidth', self.pixel_BW_is_input())
        self.set_visibility('FOV_bandwidth', self.FOV_BW_is_input())
        self.set_visibility('FW_shift', self.FW_shift_is_input())
        for voxel_size_param in ['voxel_F', 'voxel_P', 'recon_voxel_F', 'recon_voxel_P']:
            self.set_visibility(voxel_size_param, self.voxel_size_is_input())
        for matrix_param in ['matrix_F', 'matrix_P', 'recon_matrix_F', 'recon_matrix_P']:
            self.set_visibility(matrix_param, self.matrix_is_input())
        self.set_visibility('partial_Fourier', not self.from_graph('is_radial'))
        self.set_visibility('frequency_direction', not self.from_graph('is_radial'))
        self.set_visibility('phase_oversampling', not self.from_graph('is_radial'))
        self.set_visibility('radial_oversampling', self.from_graph('is_radial'))
        self.set_visibility('TI', self.from_graph('sequence_type') == 'Inversion Recovery')
        self.set_visibility('FA', self.from_graph('sequence_type') == 'Spoiled Gradient Echo')
        self.set_visibility('turbo_factor', not self.from_graph('is_gradient_echo'))
        if self.from_graph('is_gradient_echo'):
            self.set_param('turbo_factor', 1)
        self.set_visibility('homodyne', (self.from_graph('num_blank_lines') > 0 and not self.from_graph('is_radial')))
        self.set_visibility('apodization_alpha', self.from_graph('do_apodize'))

    def update_bounds(self):
        if self.from_graph('sequence_type') == 'Inversion Recovery':
            self.set_param_bounds('TI', maxval=self.from_graph('max_TI'))
        self.set_param_bounds('slice_thickness', minval=self.from_graph('min_slice_thickness'))
        if self.pixel_BW_is_input():
            self.set_param_bounds('pixel_bandwidth', minval=self.from_graph('pixel_bandwidth_bounds').min, maxval=self.from_graph('pixel_bandwidth_bounds').max)
        if self.FOV_BW_is_input():
            self.set_param_bounds('FOV_bandwidth', minval=convert.pixel_BW_to_FOV_BW(self.from_graph('pixel_bandwidth_bounds').min, self.from_graph('matrix_F')), maxval=convert.pixel_BW_to_FOV_BW(self.from_graph('pixel_bandwidth_bounds').max, self.from_graph('matrix_F')))
        if self.FW_shift_is_input():
            self.set_param_bounds('FW_shift', minval=convert.pixel_BW_to_shift(self.from_graph('pixel_bandwidth_bounds').max, self.from_graph('field_strength')), maxval=convert.pixel_BW_to_shift(self.from_graph('pixel_bandwidth_bounds').min, self.from_graph('field_strength')))
        if self.matrix_is_input():
            self.set_param_bounds('matrix_F', minval=self.from_graph('matrix_F_bounds').min, maxval=self.from_graph('matrix_F_bounds').max)
            self.set_param_bounds('matrix_P', minval=self.from_graph('matrix_P_bounds').min, maxval=self.from_graph('matrix_P_bounds').max)
            self.set_param_bounds('recon_matrix_F', minval=self.from_graph('recon_matrix_F_bounds').min, maxval=self.from_graph('recon_matrix_F_bounds').max)
            self.set_param_bounds('recon_matrix_P', minval=self.from_graph('recon_matrix_P_bounds').min, maxval=self.from_graph('recon_matrix_P_bounds').max)
        self.set_param_bounds('FOV_F', minval=self.from_graph('FOV_F_bounds').min, maxval=self.from_graph('FOV_F_bounds').max)
        self.set_param_bounds('FOV_P', minval=self.from_graph('FOV_P_bounds').min, maxval=self.from_graph('FOV_P_bounds').max)
        if self.voxel_size_is_input():
            self.set_param_bounds('voxel_F', minval=self.from_graph('FOV_F')/self.from_graph('matrix_F_bounds').max, maxval=self.from_graph('FOV_F')/self.from_graph('matrix_F_bounds').min)
            self.set_param_bounds('voxel_P', minval=self.from_graph('FOV_P')/self.from_graph('matrix_P_bounds').max, maxval=self.from_graph('FOV_P')/self.from_graph('matrix_P_bounds').min)
            self.set_param_bounds('recon_voxel_F', minval=self.from_graph('FOV_F')/self.from_graph('recon_matrix_F_bounds').max, maxval=self.from_graph('FOV_F')/self.from_graph('recon_matrix_F_bounds').min)
            self.set_param_bounds('recon_voxel_P', minval=self.from_graph('FOV_P')/self.from_graph('recon_matrix_P_bounds').max, maxval=self.from_graph('FOV_P')/self.from_graph('recon_matrix_P_bounds').min)
    
        self.update_turbo_factor_bounds()
        self.update_EPI_factor_bounds()

        self.input.param.reference_tissue.objects = self.from_graph('tissues')
        if 'object' in self.from_graph('trigger_nodes') and 'reference_tissue' not in self.from_graph('trigger_nodes'):
            self.input.reference_tissue = self.from_graph('tissues')[0]

        self.set_param_bounds('TR', minval=self.from_graph('min_TR'))
        self.set_param_bounds('TE', minval=self.from_graph('min_TE'), maxval=self.from_graph('max_TE'))
        self.input.param.shot.bounds = (0, self.from_graph('num_shots') - 1)

        self.update_x_y_labels()
        self.update_shot_label()

    def update_turbo_factor_bounds(self):
        # turbo_factor must equal 1 when the EPI_factor is even
        if not self.input.EPI_factor%2:
            self.input.param.turbo_factor.bounds = (1, 1)
            self.input.param.turbo_factor.constant = True
            return
        self.input.param.turbo_factor.bounds = (1, min(self.from_graph('max_turbo_factor'), PARAMS['turbo_factor'].bounds[-1]))
        self.input.param.turbo_factor.constant = False

    def update_EPI_factor_bounds(self):
        self.set_param_bounds('EPI_factor', maxval=self.from_graph('max_EPI_factor'))
        # EPI_factor must be odd for turbo spin echo (GRASE)
        if self.input.turbo_factor > 1:
            self.input.param.EPI_factor.objects = [v for v in self.input.param.EPI_factor.objects if v%2]

    def update_x_y_labels(self):
        frequency_direction = self.from_graph('frequency_direction')
        for p in ['FOV_F', 'FOV_P', 'matrix_F', 'matrix_P', 'recon_matrix_F', 'recon_matrix_P']:
            par = self.input.param[p]
            if (' y' in par.label) and (('_F' in par.name and frequency_direction=='left-right') or
                                        ('_P' in par.name and frequency_direction=='anterior-posterior')):
                par.label = par.label.replace(' y', ' x')
            elif (' x' in par.label) and (('_P' in par.name and frequency_direction=='left-right') or
                                        ('_F' in par.name and frequency_direction=='anterior-posterior')):
                par.label = par.label.replace(' x', ' y')

    def update_shot_label(self):
        self.shot_label = 'shot' if not self.from_graph('is_radial') else 'spoke' if (self.from_graph('EPI_factor') * self.from_graph('turbo_factor') == 1) else 'blade'
        self.input.param.shot.label = f'Displayed {self.shot_label}'

    def update_params(self):
        self.passive_updates = True

        for par in self.input_nodes():
            self.set_param(par, self.from_graph(par))
        self.set_param('FOV_bandwidth', convert.pixel_BW_to_FOV_BW(self.from_graph('pixel_bandwidth'), self.from_graph('matrix_F')))
        self.set_param('FW_shift', convert.pixel_BW_to_shift(self.from_graph('pixel_bandwidth'), self.from_graph('field_strength')))
        for dir in ['F', 'P']:
            for prefix in ['', 'recon_']:
                self.set_param(f'{prefix}voxel_{dir}', self.from_graph(f'FOV_{dir}') / self.from_graph(f'{prefix}matrix_{dir}'))
        self.update_trajectory_lables()

        self.passive_updates = False

        if 'object' in self.from_graph('trigger_nodes'):
            if self.from_graph('FOV_F') < self.from_graph('phantom_object')['support'][self.from_graph('freq_dir')]:
                self.set_param('FOV_F', self.from_graph('phantom_object')['support'][self.from_graph('freq_dir')], mode='ceil')
            if self.from_graph('FOV_P') < self.from_graph('phantom_object')['support'][self.from_graph('phase_dir')]:
                self.set_param('FOV_P', self.from_graph('phantom_object')['support'][self.from_graph('phase_dir')], mode='ceil')

    def update_trajectory_lables(self):
        # Label radial trajectory 'Radial' or 'PROPELLER' depending on nLines per shot
        invalid, updated = ('PROPELLER', 'Radial') if (self.from_graph('EPI_factor') * self.from_graph('turbo_factor') == 1) else ('Radial', 'PROPELLER')
        if self.input.trajectory == invalid:
            self.input.param.trajectory.objects = PARAMS['trajectory'].objects
            self.input.trajectory = updated
        self.input.param.trajectory.objects = [t for t in PARAMS['trajectory'].objects if t != invalid]

    def update_info_params(self):
        self.gui.FW_shift = formatting.FW_shift(self.input.FW_shift)
        self.gui.relative_SNR = formatting.relative_SNR(self.from_graph('SNR') / self.reference_SNR)
        for par in ['spoke_angle', 'num_shots', 'scantime', 'pixel_bandwidth']:
            setattr(self.gui, par, getattr(formatting, par)(self.from_graph(par)))

    def update_plots(self):
        self.gui.image = self.from_graph('annotated_image')
        self.gui.kspace = self.from_graph('kspace')
        hover = self.from_graph('hover_manager')
        hover.k_trajectory = self.from_graph('k_trajectory')
        for board in ['frequency', 'phase', 'RF', 'signal']:
            hover.objects[board] = self.from_graph(f'{board}_objects')
        self.gui.sequence_plot = self.from_graph('sequence_plot')

    def update_rec_acq_ratio(self):
        self.rec_acq_ratio_F = self.from_graph('recon_matrix_F') / self.from_graph('matrix_F')
        self.rec_acq_ratio_P = self.from_graph('recon_matrix_P') / self.from_graph('matrix_P')