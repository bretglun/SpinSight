
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
def matrix_F(FOV_F, matrix_F_prescribed, isotropic_voxel_size, FOV_P, matrix_P):
    if isotropic_voxel_size:
        voxel = FOV_P / matrix_P
        return int(np.round(FOV_F / voxel))
    return matrix_F_prescribed


@Graph.node()
def EPI_factor(EPI_factor_prescribed, turbo_factor):
    if turbo_factor > 1 and not EPI_factor_prescribed % 2:
        return EPI_factor_prescribed + 1 # EPI_factor must be odd for GRASE
    return EPI_factor_prescribed


@Graph.node()
def shot(shot_prescribed, num_shots):
    return min(shot_prescribed, num_shots)


@Graph.node()
def reference_tissue(reference_tissue_prescribed, tissues):
    if reference_tissue_prescribed in tissues:
        return reference_tissue_prescribed
    return tissues[0]