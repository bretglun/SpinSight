from spinsight import recon
from spinsight.DAG import Graph
import numpy as np
import copy


@Graph.node()
def zerofilled_kspace(apodized_kspace, oversampled_recon_matrix):
    return recon.zerofill(apodized_kspace, oversampled_recon_matrix)


@Graph.node()
def oversampled_recon_matrix(recon_matrix, full_k_matrix, matrix):
    oversampled_recon_matrix = copy.deepcopy(recon_matrix)
    for dim in range(2):
        oversampled_recon_matrix[dim] = int(np.round(recon_matrix[dim] * full_k_matrix[dim] / matrix[dim]))
    return oversampled_recon_matrix


@Graph.node()
def apodized_kspace(full_kspace, do_apodize, apodization_alpha):
    apodized_kspace = copy.deepcopy(full_kspace)
    if do_apodize: 
        apodized_kspace *= recon.radial_Tukey(apodization_alpha, full_kspace.shape)
    return apodized_kspace


@Graph.node()
def full_kspace(num_blank_lines, is_radial, gridded_kspace, phase_dir, homodyne, num_phase_encodes_full):
    if (num_blank_lines == 0 or is_radial):
        return np.copy(gridded_kspace)
    shape_unsampled = tuple(num_blank_lines if dim==phase_dir else n for dim, n in enumerate(gridded_kspace.shape))
    full_kspace = np.append(gridded_kspace, np.zeros(shape_unsampled), axis=phase_dir) # zerofill
    if homodyne and (num_blank_lines > 0):
        full_kspace *= recon.homodyne_weights(num_phase_encodes_full, num_blank_lines, phase_dir) # pre-weighting
        full_kspace += np.conjugate(np.flip(full_kspace))
    return full_kspace


@Graph.node()
def full_k_matrix(full_kspace):
    return full_kspace.shape


@Graph.node()
def gridded_kspace(k_grid_axes, is_radial, measured_kspace, k_samples, FOV, matrix):
    grid_shape = tuple(len(k_grid_axes[dim]) for dim in range(2))
    if not is_radial:
        return measured_kspace.reshape(grid_shape)
    samples = k_samples * FOV / matrix
    return recon.grid(measured_kspace, grid_shape, samples)