
from spinsight import convert, params
from spinsight.DAG import Graph
from spinsight.params import PARAMS
import numpy as np


@Graph.node()
def TR(TR_ui, min_TR):
    min_TR = params.snap(min_TR, PARAMS['TR_ui'].objects.values(), mode='ceil')
    return max(TR_ui, min_TR)


@Graph.node()
def TE(TE_ui, min_TE):
    min_TE = params.snap(min_TE, PARAMS['TE_ui'].objects.values(), mode='ceil')
    return max(TE_ui, min_TE)


@Graph.node()
def pixel_bandwidth(FOV_BW_is_input, FOV_bandwidth, matrix_F, FW_shift_is_input, FW_shift, field_strength, pixel_bandwidth_ui):
    if FOV_BW_is_input:
        return convert.FOV_BW_to_pixel_BW(FOV_bandwidth, matrix_F)
    elif FW_shift_is_input:
        return convert.shift_to_pixel_BW(FW_shift, field_strength)
    return pixel_bandwidth_ui


@Graph.node()
def isotropic_voxel_size(is_radial):
    return is_radial


@Graph.node()
def matrix_F(voxel_size_is_input, FOV_F, voxel_F, matrix_F_ui, isotropic_voxel_size, trigger_node, voxel_P, FOV_P, matrix_P_ui):
    if isotropic_voxel_size and trigger_node in ['matrix_P_ui', 'voxel_P']:
        voxel = voxel_P if voxel_size_is_input else FOV_P / matrix_P_ui
        return int(np.round(FOV_F / voxel))
    return int(np.round(FOV_F / voxel_F)) if voxel_size_is_input else matrix_F_ui


@Graph.node()
def matrix_P(voxel_size_is_input, FOV_P, voxel_P, matrix_P_ui, isotropic_voxel_size, trigger_node, voxel_F, FOV_F, matrix_F_ui):
    if isotropic_voxel_size and trigger_node in ['matrix_F_ui', 'voxel_F', 'trajectory']:
        voxel = voxel_F if voxel_size_is_input else FOV_F / matrix_F_ui
        return int(np.round(FOV_P / voxel))
    return int(np.round(FOV_P / voxel_P)) if voxel_size_is_input else matrix_P_ui


@Graph.node()
def keep_rec_acq_ratio(trigger_node):
    return trigger_node not in ['recon_voxel_P', 'recon_voxel_F', 'recon_matrix_P_ui', 'recon_matrix_F_ui']


@Graph.node()
def recon_matrix_F(keep_rec_acq_ratio, matrix_F, rec_acq_ratio_F, voxel_size_is_input, FOV_F, recon_voxel_F, recon_matrix_F_ui):
    if keep_rec_acq_ratio:
        return int(matrix_F * rec_acq_ratio_F)
    if voxel_size_is_input:
        return int(np.round(FOV_F / recon_voxel_F))
    return recon_matrix_F_ui


@Graph.node()
def recon_matrix_P(keep_rec_acq_ratio, matrix_P, rec_acq_ratio_P, voxel_size_is_input, FOV_P, recon_voxel_P, recon_matrix_P_ui):
    if keep_rec_acq_ratio:
        return int(matrix_P * rec_acq_ratio_P)
    if voxel_size_is_input:
        return int(np.round(FOV_P / recon_voxel_P))
    return recon_matrix_P_ui


@Graph.node()
def shot(shot_ui, num_shots):
    return min(shot_ui, num_shots) - 1