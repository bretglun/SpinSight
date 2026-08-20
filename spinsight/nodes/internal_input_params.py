
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
def pixel_bandwidth(FOV_BW_is_input, FOV_bandwidth, matrix_F, FW_shift_is_input, FW_shift_ui, field_strength, pixel_bandwidth_ui):
    if FOV_BW_is_input:
        return convert.FOV_BW_to_pixel_BW(FOV_bandwidth, matrix_F)
    elif FW_shift_is_input:
        return convert.shift_to_pixel_BW(FW_shift_ui, field_strength)
    return pixel_bandwidth_ui


@Graph.node()
def FW_shift(pixel_bandwidth, field_strength):
    return convert.pixel_BW_to_shift(pixel_bandwidth, field_strength)


@Graph.node()
def isotropic_voxel_size(is_radial):
    return is_radial


@Graph.node()
def matrix_F(FOV_F, matrix_F_ui, isotropic_voxel_size, trigger_nodes, FOV_P, matrix_P_ui):
    if isotropic_voxel_size and (trigger_nodes & {'matrix_P_ui'}):
        voxel = FOV_P / matrix_P_ui
        return int(np.round(FOV_F / voxel))
    return matrix_F_ui


@Graph.node()
def matrix_P(FOV_P, matrix_P_ui, isotropic_voxel_size, trigger_nodes, FOV_F, matrix_F_ui):
    if isotropic_voxel_size and (trigger_nodes & {'matrix_F_ui', 'trajectory'}):
        voxel = FOV_F / matrix_F_ui
        return int(np.round(FOV_P / voxel))
    return matrix_P_ui


@Graph.node()
def shot(shot_ui, num_shots):
    return min(shot_ui, num_shots) - 1