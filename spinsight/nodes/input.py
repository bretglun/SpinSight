from spinsight.DAG import Graph
from spinsight.params import PARAMS


@Graph.node()
def object():
    return PARAMS['object'].default


@Graph.node()
def field_strength():
    return PARAMS['field_strength'].default


@Graph.node()
def parameter_style():
    return PARAMS['parameter_style'].default


@Graph.node()
def min_voxel_size():
    return PARAMS['min_voxel_size'].default


@Graph.node()
def noise_gain():
    return PARAMS['noise_gain'].default


@Graph.node()
def sequence_type():
    return PARAMS['sequence_type'].default


@Graph.node()
def pixel_bandwidth():
    return PARAMS['pixel_bandwidth'].default


@Graph.node()
def FOV_bandwidth():
    return PARAMS['FOV_bandwidth'].default


@Graph.node()
def NSA():
    return PARAMS['NSA'].default


@Graph.node()
def partial_Fourier():
    return PARAMS['partial_Fourier'].default


@Graph.node()
def turbo_factor():
    return PARAMS['turbo_factor'].default


@Graph.node()
def EPI_factor():
    return PARAMS['EPI_factor'].default


@Graph.node()
def FatSat():
    return PARAMS['FatSat'].default


@Graph.node()
def TR_ui():
    return PARAMS['TR_ui'].default


@Graph.node()
def TE_ui():
    return PARAMS['TE_ui'].default


@Graph.node()
def TI():
    return PARAMS['TI'].default


@Graph.node()
def FA():
    return PARAMS['FA'].default


@Graph.node()
def trajectory():
    return PARAMS['trajectory'].default


@Graph.node()
def frequency_direction():
    return PARAMS['frequency_direction'].default


@Graph.node()
def FOV_P():
    return PARAMS['FOV_P'].default


@Graph.node()
def FOV_F():
    return PARAMS['FOV_F'].default


@Graph.node()
def phase_oversampling():
    return PARAMS['phase_oversampling'].default


@Graph.node()
def radial_oversampling():
    return PARAMS['radial_oversampling'].default


@Graph.node()
def matrix_P_ui():
    return PARAMS['matrix_P_ui'].default


@Graph.node()
def matrix_F_ui():
    return PARAMS['matrix_F_ui'].default


@Graph.node()
def recon_matrix_P():
    return PARAMS['recon_matrix_P'].default


@Graph.node()
def recon_matrix_F():
    return PARAMS['recon_matrix_F'].default


@Graph.node()
def slice_thickness():
    return PARAMS['slice_thickness'].default


@Graph.node()
def radial_FOV_oversampling():
    return PARAMS['radial_FOV_oversampling'].default


@Graph.node()
def show_FOV():
    return PARAMS['show_FOV'].default


@Graph.node()
def reference_tissue():
    return PARAMS['reference_tissue'].default


@Graph.node()
def image_type():
    return PARAMS['image_type'].default


@Graph.node()
def show_processed_kspace():
    return PARAMS['show_processed_kspace'].default


@Graph.node()
def kspace_exponent():
    return PARAMS['kspace_exponent'].default


@Graph.node()
def kspace_type():
    return PARAMS['kspace_type'].default


@Graph.node()
def homodyne():
    return PARAMS['homodyne'].default


@Graph.node()
def do_apodize():
    return PARAMS['do_apodize'].default


@Graph.node()
def apodization_alpha():
    return PARAMS['apodization_alpha'].default


@Graph.node()
def do_zerofill():
    return PARAMS['do_zerofill'].default


@Graph.node()
def shot_ui():
    return PARAMS['shot_ui'].default


@Graph.node()
def signal_exponent():
    return PARAMS['signal_exponent'].default


@Graph.node()
def constant_voxel_bounds():
    return False


@Graph.node()
def constant_FOV_BW_bounds():
    return False