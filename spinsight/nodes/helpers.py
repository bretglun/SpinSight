from spinsight.DAG import Graph
from spinsight.params import PARAMS


@Graph.node()
def is_radial(trajectory):
    return trajectory in ['Radial', 'PROPELLER']


@Graph.node()
def is_gradient_echo(sequence_type):
    return 'Gradient Echo' in sequence_type


@Graph.node()
def freq_dir(frequency_direction, is_radial):
    return PARAMS['frequency_direction'].objects.index(frequency_direction) if not is_radial else 1


@Graph.node()
def phase_dir(freq_dir):
    return 1 - freq_dir


@Graph.node()
def FOV(FOV_F, FOV_P, freq_dir):
    return [FOV_P, FOV_F] if freq_dir else [FOV_F, FOV_P]


@Graph.node()
def matrix(matrix_F, matrix_P, freq_dir):
    return [matrix_P, matrix_F] if freq_dir else [matrix_F, matrix_P]


@Graph.node()
def recon_matrix(recon_matrix_P, recon_matrix_F, freq_dir, do_zerofill, matrix):
    return ([recon_matrix_P, recon_matrix_F] if freq_dir else [recon_matrix_F, recon_matrix_P]) if do_zerofill else matrix


@Graph.node()
def pixel_BW_is_input(parameter_style):
    return 'PIXEL BW' in parameter_style.upper()


@Graph.node()
def FOV_BW_is_input(parameter_style):
    return 'FOV BW' in parameter_style.upper()


@Graph.node()
def FW_shift_is_input(parameter_style):
    return 'FAT/WATER SHIFT' in parameter_style.upper()


@Graph.node()
def matrix_is_input(parameter_style):
    return 'MATRIX' in parameter_style.upper()


@Graph.node()
def voxel_size_is_input(parameter_style):
    return 'VOXEL SIZE' in parameter_style.upper()