from spinsight import convert
from spinsight.DAG import Graph
from spinsight.params import PARAMS
from spinsight.constants import ACTION


@Graph.node(action=ACTION.VALUE)
def set_spoke_angle(controller, spoke_angle):
    controller.spoke_angle = spoke_angle


@Graph.node(action=ACTION.VALUE)
def set_num_shots(controller, num_shots):
    controller.num_shots = num_shots


@Graph.node(action=ACTION.VALUE)
def set_relative_SNR(controller, relative_SNR):
    controller.relative_SNR = relative_SNR


@Graph.node(action=ACTION.VALUE)
def set_scantime(controller, scantime):
    controller.scantime = scantime


@Graph.node(action=ACTION.VALUE)
def set_pixel_bandwidth(controller, pixel_BW_is_input, pixel_bandwidth):
    if not pixel_BW_is_input:
        controller.set_param('pixel_bandwidth_ui', pixel_bandwidth)


@Graph.node(action=ACTION.VALUE)
def set_FOV_bandwidth(controller, FOV_BW_is_input, pixel_bandwidth, matrix_F):
    if not FOV_BW_is_input:
        controller.set_param('FOV_bandwidth', convert.pixel_BW_to_FOV_BW(pixel_bandwidth, matrix_F))


@Graph.node(action=ACTION.VALUE)
def set_FW_shift(controller, FW_shift_is_input, pixel_bandwidth, field_strength):
    if not FW_shift_is_input:
        controller.set_param('FW_shift', convert.pixel_BW_to_shift(pixel_bandwidth, field_strength))


@Graph.node(action=ACTION.VALUE)
def set_matrix_F(controller, matrix_is_input, isotropic_voxel_size, matrix_F):
    if not matrix_is_input or isotropic_voxel_size:
        controller.set_param('matrix_F_ui', matrix_F)


@Graph.node(action=ACTION.VALUE)
def set_matrix_P(controller, matrix_is_input, isotropic_voxel_size, matrix_P):
    if not matrix_is_input or isotropic_voxel_size:
        controller.set_param('matrix_P_ui', matrix_P)


@Graph.node(action=ACTION.VALUE)
def set_recon_matrix_F(controller, matrix_is_input, keep_rec_acq_ratio, recon_matrix_F):
    if not matrix_is_input or keep_rec_acq_ratio:
        controller.set_param('recon_matrix_F_ui', recon_matrix_F)


@Graph.node(action=ACTION.VALUE)
def set_recon_matrix_P(controller, matrix_is_input, keep_rec_acq_ratio, recon_matrix_P):
    if not matrix_is_input or keep_rec_acq_ratio:
        controller.set_param('recon_matrix_P_ui', recon_matrix_P)


@Graph.node(action=ACTION.VALUE)
def set_voxel_F(controller, voxel_size_is_input, isotropic_voxel_size, FOV_F, matrix_F):
    if not voxel_size_is_input or isotropic_voxel_size:
        controller.set_param('voxel_F', FOV_F / matrix_F)


@Graph.node(action=ACTION.VALUE)
def set_voxel_P(controller, voxel_size_is_input, isotropic_voxel_size, FOV_P, matrix_P):
    if not voxel_size_is_input or isotropic_voxel_size:
        controller.set_param('voxel_P', FOV_P / matrix_P)


@Graph.node(action=ACTION.VALUE)
def set_recon_voxel_F(controller, voxel_size_is_input, keep_rec_acq_ratio, FOV_F, recon_matrix_F):
    if not voxel_size_is_input or keep_rec_acq_ratio:
        controller.set_param('recon_voxel_F', FOV_F / recon_matrix_F)


@Graph.node(action=ACTION.VALUE)
def set_recon_voxel_P(controller, voxel_size_is_input, keep_rec_acq_ratio, FOV_P, recon_matrix_P):
    if not voxel_size_is_input or keep_rec_acq_ratio:
        controller.set_param('recon_voxel_P', FOV_P / recon_matrix_P)


@Graph.node(action=ACTION.VALUE)
def set_TR_and_bounds(controller, min_TR, TR):
    controller.set_param_bounds('TR_ui', minval=min_TR)
    controller.set_param('TR_ui', TR)


@Graph.node(action=ACTION.VALUE)
def set_TE_and_bounds(controller, min_TE, max_TE, TE):
    controller.set_param_bounds('TE_ui', minval=min_TE, maxval=max_TE)
    controller.set_param('TE_ui', TE)


@Graph.node(action=ACTION.VALUE)
def set_shot_and_bounds(controller, num_shots, shot):
    controller.input.param.shot_ui.bounds = (1, num_shots)
    controller.set_param('shot_ui', shot + 1)


@Graph.node(action=ACTION.VALUE)
def set_trajectory_objects(controller, EPI_factor, turbo_factor):
    # Label radial trajectory 'Radial' or 'PROPELLER' depending on nLines per shot
    controller.input.param.trajectory.objects = PARAMS['trajectory'].objects
    invalid, updated = ('PROPELLER', 'Radial') if (EPI_factor * turbo_factor == 1) else ('Radial', 'PROPELLER')
    if controller.input.trajectory == invalid:
        controller.input.trajectory = updated
    controller.input.param.trajectory.objects = [t for t in PARAMS['trajectory'].objects if t != invalid]


@Graph.node(action=ACTION.INVISIBLE)
def set_rec_acq_ratio_F(controller, recon_matrix_F, matrix_F):
    controller.rec_acq_ratio_F = recon_matrix_F / matrix_F


@Graph.node(action=ACTION.INVISIBLE)
def set_rec_acq_ratio_P(controller, recon_matrix_P, matrix_P):
    controller.rec_acq_ratio_P = recon_matrix_P / matrix_P