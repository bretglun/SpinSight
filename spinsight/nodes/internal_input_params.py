
from spinsight import convert, params
from spinsight.DAG import Graph
from spinsight.params import PARAMS
import numpy as np


@Graph.node()
def TR(TR_prescribed, min_TR):
    min_TR = params.snap(min_TR, PARAMS['TR'].objects.values(), mode='ceil')
    return max(TR_prescribed, min_TR)


@Graph.node()
def TE(TE_prescribed, min_TE):
    min_TE = params.snap(min_TE, PARAMS['TE'].objects.values(), mode='ceil')
    return max(TE_prescribed, min_TE)


@Graph.node()
def isotropic_voxel_size(is_radial):
    return is_radial


@Graph.node()
def matrix_F(FOV_F, matrix_F_prescribed, isotropic_voxel_size, trigger_nodes, FOV_P, matrix_P_prescribed):
    if isotropic_voxel_size and (trigger_nodes & {'matrix_P'}):
        voxel = FOV_P / matrix_P_prescribed
        return int(np.round(FOV_F / voxel))
    return matrix_F_prescribed


@Graph.node()
def matrix_P(FOV_P, matrix_P_prescribed, isotropic_voxel_size, trigger_nodes, FOV_F, matrix_F_prescribed):
    if isotropic_voxel_size and (trigger_nodes & {'matrix_F', 'trajectory'}):
        voxel = FOV_F / matrix_F_prescribed
        return int(np.round(FOV_P / voxel))
    return matrix_P_prescribed


@Graph.node()
def shot(shot_prescribed, num_shots):
    return min(shot_prescribed, num_shots) - 1


@Graph.node()
def reference_tissue(reference_tissue_prescribed, tissues):
    if reference_tissue_prescribed in tissues:
        return reference_tissue_prescribed
    return tissues[0]