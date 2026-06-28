from spinsight import convert
from spinsight.DAG import Graph
from spinsight.params import PARAMS
from spinsight.constants import ACTION


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


@Graph.node(action=ACTION.VALUE)
def set_shot_and_bounds(simulator, num_shots, shot):
    simulator.param.shot_ui.bounds = (1, num_shots)
    simulator.set_param(simulator.param.shot_ui, shot + 1)


@Graph.node(action=ACTION.VALUE)
def set_trajectory_objects(simulator, EPI_factor, turbo_factor):
    # Label radial trajectory 'Radial' or 'PROPELLER' depending on nLines per shot
    simulator.param.trajectory.objects = PARAMS['trajectory'].objects
    invalid, updated = ('PROPELLER', 'Radial') if (EPI_factor * turbo_factor == 1) else ('Radial', 'PROPELLER')
    if simulator.trajectory == invalid:
        simulator.trajectory = updated
    simulator.param.trajectory.objects = [t for t in PARAMS['trajectory'].objects if t != invalid]