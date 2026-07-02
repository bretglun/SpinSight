
from spinsight.DAG import Graph
from spinsight.constants import ACTION


@Graph.node(action=ACTION.VISIBILITY)
def set_pixel_bandwidth_visibility(controller, pixel_BW_is_input):
    controller.set_visibility('pixel_bandwidth_ui', pixel_BW_is_input)


@Graph.node(action=ACTION.VISIBILITY)
def set_FOV_bandwidth_visibility(controller, FOV_BW_is_input):
    controller.set_visibility('FOV_bandwidth', FOV_BW_is_input)


@Graph.node(action=ACTION.VISIBILITY)
def set_FW_shift_visibility(controller, FW_shift_is_input):
    controller.set_visibility('FW_shift', FW_shift_is_input)


@Graph.node(action=ACTION.VISIBILITY)
def set_voxel_size_visibility(controller, voxel_size_is_input):
    for voxel_size_param in ['voxel_F', 'voxel_P', 'recon_voxel_F', 'recon_voxel_P']:
        controller.set_visibility(voxel_size_param, voxel_size_is_input)


@Graph.node(action=ACTION.VISIBILITY)
def set_matrix_visibility(controller, matrix_is_input):
    for matrix_param in ['matrix_F_ui', 'matrix_P_ui', 'recon_matrix_F_ui', 'recon_matrix_P_ui']:
        controller.set_visibility(matrix_param, matrix_is_input)


@Graph.node(action=ACTION.VISIBILITY)
def set_partial_Fourier_visibility(controller, is_radial):   
    controller.set_visibility('partial_Fourier', not is_radial)


@Graph.node(action=ACTION.VISIBILITY)
def set_frequency_direction_visibility(controller, is_radial):
    controller.set_visibility('frequency_direction', not is_radial)


@Graph.node(action=ACTION.VISIBILITY)
def set_phase_oversampling_visibility(controller, is_radial):
    controller.set_visibility('phase_oversampling', not is_radial)


@Graph.node(action=ACTION.VISIBILITY)
def set_radial_oversampling_visibility(controller, is_radial):
    controller.set_visibility('radial_oversampling', is_radial)


@Graph.node(action=ACTION.VISIBILITY)
def set_TI_visibility(controller, sequence_type):
    controller.set_visibility('TI', sequence_type == 'Inversion Recovery')


@Graph.node(action=ACTION.VISIBILITY)
def set_FA_visibility(controller, sequence_type):
    controller.set_visibility('FA', sequence_type == 'Spoiled Gradient Echo')


@Graph.node(action=ACTION.VISIBILITY)
def set_turbo_factor_visibility(controller, is_gradient_echo):
    visible = not is_gradient_echo
    controller.set_visibility('turbo_factor', visible)
    if not visible:
        controller.set_param('turbo_factor', 1)


@Graph.node(action=ACTION.VISIBILITY)
def set_homodyne_visibility(controller, num_blank_lines, is_radial):
    controller.set_visibility('homodyne', (num_blank_lines > 0 and not is_radial))


@Graph.node(action=ACTION.VISIBILITY)
def set_apodization_alpha_visibility(controller, do_apodize):
    controller.set_visibility('apodization_alpha', do_apodize)