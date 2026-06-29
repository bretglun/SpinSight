
from spinsight.DAG import Graph
from spinsight.constants import ACTION


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
        simulator.set_param('partial_Fourier', 1)


@Graph.node(action=ACTION.VISIBILITY)
def set_frequency_direction_visibility(simulator, is_radial):
    simulator.set_visibility('frequency_direction', not is_radial)


@Graph.node(action=ACTION.VISIBILITY)
def set_phase_oversampling_visibility(simulator, is_radial):
    visible = not is_radial
    simulator.set_visibility('phase_oversampling', visible)
    if not visible:
        simulator.set_param('phase_oversampling', 1)


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
        simulator.set_param('turbo_factor', 1)


@Graph.node(action=ACTION.VISIBILITY)
def set_homodyne_visibility(simulator, num_blank_lines, is_radial):
    simulator.set_visibility('homodyne', (num_blank_lines > 0 and not is_radial))


@Graph.node(action=ACTION.VISIBILITY)
def set_apodization_alpha_visibility(simulator, do_apodize):
    simulator.set_visibility('apodization_alpha', do_apodize)