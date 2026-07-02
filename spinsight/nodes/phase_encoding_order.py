from spinsight.DAG import Graph
from spinsight.nodes.sequence_timing import get_readtrain_spacing, min_readtrain_spacing
import numpy as np
import warnings


@Graph.node()
def pe_table(num_sampled_phase_encodes, num_segm, num_shots, EPI_factor, turbo_factor, num_sym_segm, k0_rf_echo_index, reverse_linear_order):
    if EPI_factor == 1: # (turbo) spin echo
        segment_order = flexible_segment_order(turbo_factor, num_sym_segm, k0_rf_echo_index)
    else: # EPI and GRASE
        segment_order = range(turbo_factor) # linear segment order
    order = -1 if reverse_linear_order else 1
    view_order = np.array(range(EPI_factor))[None, ::order] * turbo_factor + np.array(segment_order)[::order, None]
    num_lines_per_segment = int(num_sampled_phase_encodes / num_segm)
    lines = [shot % num_lines_per_segment for shot in range(num_shots)]
    return num_lines_per_segment * view_order + np.array(lines)[:, None, None]


def flexible_segment_order(num_segm, num_sym, k0_index):
    # temporal order of k-space segments with flexible k-space center (k0)
    
    split_k0 = not(num_sym % 2) # k-space center is between two segments

    if k0_index > (num_segm - 1) // 2:
        return flexible_segment_order(num_segm, num_sym, num_segm - 1 - k0_index - split_k0)[::-1]
    
    num_sym_linear = min(num_sym, 2 * k0_index + 1 + split_k0)
    num_sym_pivoting = num_sym - num_sym_linear
    num_asym = num_segm - num_sym
    linear_start = num_asym + num_sym//2 - split_k0 - k0_index
    linear_end = num_segm - num_sym_pivoting//2
    linear = list(range(linear_start, linear_end)) # consecutively read (symmetric) segments
    if linear_start == num_asym:
        linear.reverse()
    pivoting = [val for pair in zip(range(linear_end, num_segm), reversed(range(num_asym, linear_start))) for val in pair]
    tail = list(range(min(linear_start, num_asym)))[::-1] # remaining (asymmetric) segments
    segment_order = linear + pivoting + tail
    return segment_order


@Graph.node()
def num_segm(turbo_factor, EPI_factor):
    # number of k-space segments
    return turbo_factor * EPI_factor


@Graph.node()
def num_sym_segm(num_segm, num_sym_lines, num_sampled_phase_encodes):
    # number of k-space segments symmetric about k0:
    num_sym_segm = num_segm * (num_sym_lines / num_sampled_phase_encodes)
    if (num_sym_segm % 2 == 0):
        return int(num_sym_segm) # k0 lies between two segments
    return int(np.round((num_sym_segm - 1) / 2)) * 2 + 1


@Graph.node()
def num_blades(num_shots, is_radial):
    return num_shots if is_radial else 1


@Graph.node()
def num_shots(matrix_P, phase_oversampling, partial_Fourier, turbo_factor, EPI_factor, is_radial, matrix, radial_oversampling):
    oversampling = radial_oversampling if is_radial else phase_oversampling
    undersampling = 1 if is_radial else partial_Fourier
    num_lines_Nyquist = max(matrix) * np.pi / 2 if is_radial else matrix_P # conservative for radial due to uniform angles
    lines_per_shot = turbo_factor * EPI_factor
    return int(np.ceil(num_lines_Nyquist * oversampling * undersampling / lines_per_shot))


@Graph.node()
def num_sampled_phase_encodes(turbo_factor, EPI_factor, num_shots, is_radial):
    # measured lines per blade
    return turbo_factor * EPI_factor * (num_shots if not is_radial else 1)


@Graph.node()
def num_phase_encodes_full(is_radial, num_sampled_phase_encodes, matrix_P, phase_oversampling):
    if is_radial:
        return num_sampled_phase_encodes # since no undersampling or phase oversampling
    # oversampling may be higher than prescribed since num_shots must be integer:
    return max(num_sampled_phase_encodes, int(np.ceil(matrix_P * phase_oversampling)))


@Graph.node()
def num_sym_lines(num_sampled_phase_encodes, num_phase_encodes_full):
    return 2 * num_sampled_phase_encodes - num_phase_encodes_full


@Graph.node()
def num_blank_lines(num_phase_encodes_full, sampling_mask):
    return num_phase_encodes_full - sum(sampling_mask)


@Graph.node()
def sampling_mask(num_phase_encodes_full, num_sampled_phase_encodes):
    sampling_mask = np.ones(num_phase_encodes_full, dtype=bool)
    # undersample by partial Fourier:
    sampling_mask[num_sampled_phase_encodes:] = False
    assert(sum(sampling_mask) == num_sampled_phase_encodes)
    return sampling_mask


@Graph.node()
def k0_segment(num_segm, num_sym_segm):
    k0_segment = num_segm - num_sym_segm // 2 - 1
    if (num_sym_segm % 2 == 0):
        return [k0_segment, k0_segment + 1] # k0 lies between two segments
    return [k0_segment]


@Graph.node()
def k0_echo_indices_linear_order(k0_segment, turbo_factor):
    # (k0_gr_echo_index, k0_rf_echo_index)
    return {(segment // turbo_factor, segment % turbo_factor) for segment in k0_segment}


@Graph.node()
def k0_echo_indices_reverse_linear_order(k0_echo_indices_linear_order, EPI_factor, turbo_factor):
    return {(EPI_factor - 1 - gr_index, turbo_factor - 1 - rf_index) for (gr_index, rf_index) in k0_echo_indices_linear_order}


@Graph.node()
def k0_rf_echo_index(k0_index):
    return k0_index[0]


@Graph.node()
def k0_gr_echo_index(k0_index):
    return k0_index[1]


@Graph.node()
def reverse_linear_order(k0_index):
    return k0_index[2]


@Graph.node()
def k0_index(k0_echo_indices_reverse_linear_order, k0_echo_indices_linear_order, is_gradient_echo, turbo_factor, TE, EPI_factor, gr_echo_spacing, min_refocusing_time, min_RF_to_readtrain_center):
    if EPI_factor == 1:
        gr_index = 0
        reverse_order = False
        if is_gradient_echo or turbo_factor == 1:
            rf_index = 0
        else: # flexible segment order
            readtrain_spacing = min_readtrain_spacing(gr_echo_spacing, gr_index, EPI_factor, is_gradient_echo, min_RF_to_readtrain_center, turbo_factor, min_refocusing_time)
            rf_index = int(np.floor(TE / readtrain_spacing)) - 1
            rf_index = min(rf_index, turbo_factor - 1)
        return (rf_index, gr_index, reverse_order)
    # choose order that minimizes readtrain spacing
    readtrain_spacings = {}
    for reverse_order, indices in [
        (True, k0_echo_indices_reverse_linear_order),
        (False, k0_echo_indices_linear_order)]:
        for (gr_index, rf_index) in indices:
            first_refocusing_time = get_first_refocusing_time(is_gradient_echo, turbo_factor, TE, EPI_factor, gr_echo_spacing, gr_index, rf_index)
            RF_to_readtrain_center = get_RF_to_readtrain_center(TE, EPI_factor, gr_echo_spacing, gr_index, rf_index, first_refocusing_time)
            if (first_refocusing_time >= min_refocusing_time) and (RF_to_readtrain_center >= min_RF_to_readtrain_center):
                readtrain_spacings[(rf_index, gr_index, reverse_order)] = first_refocusing_time + RF_to_readtrain_center
    if not readtrain_spacings:
        warnings.warn('No valid order found')
        return (0, 0, False)
    smallest_spacing = readtrain_spacings[min(readtrain_spacings, key=readtrain_spacings.get)]
    for k0_index in readtrain_spacings:
        if readtrain_spacings[k0_index] == smallest_spacing:
            return k0_index


def get_RF_to_readtrain_center(TE, num_gr_echoes, gr_echo_spacing, k0_gr_echo_index, k0_rf_echo_index, first_refocusing_time):
    first_readtrain_center = get_readtrain_spacing(TE, num_gr_echoes, gr_echo_spacing, k0_gr_echo_index, k0_rf_echo_index)
    return first_readtrain_center - first_refocusing_time


def get_first_refocusing_time(is_gradient_echo, num_rf_echoes, TE, num_gr_echoes, gr_echo_spacing, k0_gr_echo_index, k0_rf_echo_index):
    if is_gradient_echo:
        return 0
    if (num_rf_echoes == 1):
        return TE / 2
    return get_readtrain_spacing(TE, num_gr_echoes, gr_echo_spacing, k0_gr_echo_index, k0_rf_echo_index) / 2