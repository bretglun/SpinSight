from spinsight import phantom, recon
from spinsight.DAG import Graph
from spinsight.constants import GYRO, TISSUES, FAT_RESONANCES, T2_PRIM
import numpy as np
import copy


@Graph.node()
def measured_kspace(noise, kspace_comps_T2w, FatSat, PD_and_T1w):
    measured_kspace = copy.deepcopy(noise)
    for component in kspace_comps_T2w:
        if 'Fat' in component:
            tissue = component[:component.find('Fat')]
            resonance = component[component.find('Fat'):]
            ratio = FAT_RESONANCES[resonance]['ratio_with_FatSat' if FatSat else 'ratio']
            ratio *= TISSUES[tissue]['FF']
            measured_kspace += kspace_comps_T2w[component] * PD_and_T1w[resonance] * ratio
        else:
            if 'Water' in component:
                tissue = component[:component.find('Water')]
                ratio = 1 - TISSUES[tissue]['FF']    
            else:
                tissue = component
                ratio = 1.0
            measured_kspace += kspace_comps_T2w[component] * PD_and_T1w[tissue] * ratio
    return measured_kspace


@Graph.node()
def noise(k_samples, noise_std):
    sampled_matrix = k_samples.shape[:-1]
    return np.random.normal(0, noise_std, sampled_matrix) + 1j * np.random.normal(0, noise_std, sampled_matrix)


def get_PD_and_T1w(component, sequence_type, TR, TI, FA, B0):
    PD = TISSUES[component]['PD'] if 'Fat' not in component else FAT_RESONANCES[component]['PD']
    T1 = TISSUES[component]['T1'][B0] if 'Fat' not in component else FAT_RESONANCES[component]['T1'][B0]

    E1 = np.exp(-TR/T1)
    if sequence_type == 'Spin Echo':
        return PD * (1 - E1)
    elif sequence_type == 'Spoiled Gradient Echo':
        return PD * np.sin(np.radians(FA)) * (1 - E1) / (1 - np.cos(np.radians(FA)) * E1)
    elif sequence_type == 'Inversion Recovery':
        return PD * (1 - 2 * np.exp(-TI/T1) + E1)
    else:
        raise ValueError(f'Unknown sequence type: {sequence_type}')


@Graph.node()
def PD_and_T1w(sequence_type, TR, TI, FA, field_strength, tissues):
    return {component: get_PD_and_T1w(component, sequence_type, TR, TI, FA, field_strength) for component in set(tissues).union(set(FAT_RESONANCES.keys()))}


@Graph.node()
def kspace_comps_T2w(tissues, kspace_comps_thick, T2w, dephasing):
    kspace_comps = {}
    for tissue in tissues:
        if TISSUES[tissue]['FF'] == .00:
            kspace_comps[tissue] = kspace_comps_thick[tissue] * T2w[tissue]
        else: # fat containing tissues
            kspace_comps[tissue + 'Water'] = kspace_comps_thick[tissue] * T2w[tissue]
            for component in FAT_RESONANCES:
                kspace_comps[tissue + component] = kspace_comps_thick[tissue] * dephasing[component] * T2w[tissue + component]
    return kspace_comps


def get_T2w(component, time_after_excitation, time_relative_inphase, B0):
    T2 = TISSUES[component]['T2'][B0] if 'Fat' not in component else FAT_RESONANCES[component]['T2'][B0]
    T2prim = T2_PRIM
    E2 = np.exp(-np.abs(time_after_excitation)/T2)
    E2prim = np.exp(-np.abs(time_relative_inphase)/T2prim)
    return E2 * E2prim


@Graph.node()
def T2w(tissues, time_after_excitation, time_relative_inphase, field_strength):
    T2w = {}
    for tissue in tissues:
        T2w[tissue] = get_T2w(tissue, time_after_excitation, time_relative_inphase, field_strength)
        if TISSUES[tissue]['FF'] > 0: # fat containing tissues
            for component in FAT_RESONANCES:
                T2w[tissue + component] = get_T2w(component, time_after_excitation, time_relative_inphase, field_strength)
    return T2w


@Graph.node()
def dephasing(field_strength, time_relative_inphase):
    dephasing = {}
    for component, resonance in FAT_RESONANCES.items():
        dephasing[component] = np.exp(2j*np.pi * GYRO * field_strength * resonance['shift'] * time_relative_inphase * 1e-3)
    return dephasing


@Graph.node()
def time_relative_inphase(time_after_excitation, is_gradient_echo, spin_echoes, phase_dir):
    time_relative_inphase = copy.deepcopy(time_after_excitation)
    if not is_gradient_echo:
        # for spinecho, subtract Hahn echo position from time_after_excitation
        time_relative_inphase -= np.expand_dims(spin_echoes, axis=[dim for dim in range(3) if dim != phase_dir])
    return time_relative_inphase


@Graph.node()
def spin_echoes(lines_to_measure, pe_table, readtrain_spacing):
    spin_echoes = np.zeros((sum(lines_to_measure)))
    for ky in range(sum(lines_to_measure)):
        shot, rf_echo, gr_echo = np.argwhere(pe_table==ky)[0]
        spin_echoes[ky] = (rf_echo + 1) * readtrain_spacing
    return spin_echoes


@Graph.node()
def time_after_excitation(lines_to_measure, pe_table, readouts, sampling_time, freq_dir, phase_dir):
    TEs = np.zeros((sum(lines_to_measure)))
    reverse = np.zeros((sum(lines_to_measure)), dtype=bool)
    for ky in range(sum(lines_to_measure)):
        shot, rf_echo, gr_echo = np.argwhere(pe_table==ky)[0]
        TEs[ky] = readouts[rf_echo][gr_echo]['center_f']
        reverse[ky] = readouts[rf_echo][gr_echo]['area_f'] < 0
    sampling_offset = np.expand_dims(sampling_time, axis=[dim for dim in range(3) if dim != freq_dir])
    time_after_excitation = np.expand_dims(TEs, axis=[dim for dim in range(3) if dim != phase_dir]) + sampling_offset
    # EPI rowflip:
    reverse_time_after_excitation = np.flip(time_after_excitation, axis=freq_dir)
    reverse = np.expand_dims(reverse, axis=[dim for dim in range(3) if dim != phase_dir])
    reverse = reverse.repeat(len(sampling_time), axis=freq_dir)
    time_after_excitation[reverse] = reverse_time_after_excitation[reverse]
    return time_after_excitation


@Graph.node()
def sampling_time(pixel_bandwidth, k_read_axis):
    # time of sample along (positive) readout relative to (k-space) center
    half_read_duration = .5e3 / pixel_bandwidth # msec
    return np.linspace(-half_read_duration, half_read_duration, len(k_read_axis))

@Graph.node()
def noise_std(sampling_time, noise_gain, NSA, field_strength):
    dwell_time = np.diff(sampling_time[:2])[0]
    return noise_gain / np.sqrt(dwell_time * NSA) / field_strength



@Graph.node()
def kspace_comps_thick(slice_thickness, k_samples, kspace_comps):
    # Lorenzian line shape to mimic slice thickness
    blur_factor = .5
    slice_thickness_filter = slice_thickness * np.exp(-blur_factor * slice_thickness * np.sqrt(np.sum(k_samples**2, axis=-1)))
    kspace_comps_thick = {}
    for tissue in kspace_comps:
        kspace_comps_thick[tissue] = kspace_comps[tissue] * slice_thickness_filter
    return kspace_comps_thick


@Graph.node()
def kspace_comps(is_radial, phantom_object, k_grid_axes, k_samples):
    if not is_radial:
        return recon.resample_kspace_Cartesian(phantom_object, k_grid_axes, shape=k_samples.shape[:-1])
    return recon.resample_kspace(phantom_object, k_samples)


@Graph.node()
def k_grid_axes(is_radial, k_axes, FOV, matrix, phantom_object):
    if not is_radial:
        return copy.deepcopy(k_axes)
    k_grid_axes = [None, None]
    for dim in range(2):
        voxel_size = FOV[dim] / matrix[dim]
        matrix_dim = int(np.ceil(max(FOV[dim], phantom_object['support'][dim]) / voxel_size))
        k_grid_axes[dim] = recon.get_k_axis(matrix_dim, voxel_size)
    return k_grid_axes


@Graph.node()
def k_samples(k_axes, k_angles):
    k_samples = np.array(np.meshgrid(k_axes[0], k_axes[1])).T
    # rotate samples for each angle:
    rotmat = np.array([[np.cos(k_angles), -np.sin(k_angles)], 
                        [np.sin(k_angles),  np.cos(k_angles)]])
    return np.einsum('ijk,klm->ijml', k_samples, rotmat) # shape=(Nx, Ny, Nangles, 2)


@Graph.node()
def k_angles(num_blades):
    return np.linspace(0, np.pi, num_blades, endpoint=False)


@Graph.node()
def k_axes(freq_dir, phase_dir, k_read_axis, k_phase_axis, lines_to_measure):
    k_axes = [None]*2
    k_axes[freq_dir] = k_read_axis
    k_axes[phase_dir] = k_phase_axis[lines_to_measure]
    return k_axes


@Graph.node()
def k_phase_axis(is_radial, FOV_F, FOV_P, num_phase_encodes_full, matrix_P):
    if is_radial:
        voxel_size = max(FOV_F, FOV_P) / num_phase_encodes_full # corresponding to blade width
    else:
        voxel_size = FOV_P / matrix_P
    return recon.get_k_axis(num_phase_encodes_full, voxel_size)


@Graph.node()
def k_read_axis(freq_dir, FOV, matrix, is_radial, phantom_object, radial_FOV_oversampling):
    voxel_size = FOV[freq_dir] / matrix[freq_dir]
    if not is_radial:
        num_samples = matrix[freq_dir]
        # at least Nyquist sampling wrt phantom if loaded
        if FOV[freq_dir] < phantom_object['support'][freq_dir]:
            num_samples = int(np.ceil(phantom_object['support'][freq_dir] / voxel_size))
    else:
        maxFOV = max(max(phantom_object['support']), max(FOV))
        num_samples = int(np.ceil(maxFOV / voxel_size * radial_FOV_oversampling))
    return recon.get_k_axis(num_samples, voxel_size)


@Graph.node()
def tissues(phantom_object):
    return list(phantom_object['shapes'].keys())


@Graph.node()
def phantom_object(object, min_voxel_size):
    return phantom.load(object, min_voxel_size)