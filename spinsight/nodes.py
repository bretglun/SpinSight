import xarray as xr
from spinsight import constants, formatting, sequence, recon, phantom
from spinsight.DAG import Graph
from functools import partial
import numpy as np
import warnings
import copy
import holoviews as hv
from collections import namedtuple


MinMax = namedtuple('MinMax', ['min', 'max'])

### Helper functions ###

def hline(time_dim, amp_dim):
    return hv.HLine(0.0, kdims=[time_dim, amp_dim]).opts(tools=['xwheel_zoom', 'xpan', 'reset'], default_tools=[], active_tools=['xwheel_zoom', 'xpan'])


def get_T2w(component, time_after_excitation, time_relative_inphase, B0):
    T2 = constants.TISSUES[component]['T2'][B0] if 'Fat' not in component else constants.FAT_RESONANCES[component]['T2'][B0]
    T2prim = 35. # ad hoc value [msec]
    E2 = np.exp(-np.abs(time_after_excitation)/T2)
    E2prim = np.exp(-np.abs(time_relative_inphase)/T2prim)
    return E2 * E2prim


def get_PD_and_T1w(component, sequence_type, TR, TI, FA, B0):
    PD = constants.TISSUES[component]['PD'] if 'Fat' not in component else constants.FAT_RESONANCES[component]['PD']
    T1 = constants.TISSUES[component]['T1'][B0] if 'Fat' not in component else constants.FAT_RESONANCES[component]['T1'][B0]

    E1 = np.exp(-TR/T1)
    if sequence_type == 'Spin Echo':
        return PD * (1 - E1)
    elif sequence_type == 'Spoiled Gradient Echo':
        return PD * np.sin(np.radians(FA)) * (1 - E1) / (1 - np.cos(np.radians(FA)) * E1)
    elif sequence_type == 'Inversion Recovery':
        return PD * (1 - 2 * np.exp(-TI/T1) + E1)
    else:
        raise ValueError(f'Unknown sequence type: {sequence_type}')


def get_segment_order(N, Nsym, c):
    '''Returns the temporal order in which to read k-space segments for a spin-echo train

    Args:
        N: number of segments, i.e. echo train length
        Nsym: number of segments symmetric about the center of k-space
        c: index of spin echo where (the first) centermost k-space segment is read

    Returns:
        Segment indices as a temporally ordered list
    '''

    split_k0 = not(Nsym % 2) # k-space center is between two segments

    if c >= N - split_k0:
        raise ValueError('The spin echo index of (the first) centermost k-space segment is too high')
    elif c > N//2 - split_k0:
        return get_segment_order(N, Nsym, N-1-c-split_k0)[::-1]
    
    Ncon = min(2 * c + 1 + split_k0, Nsym) # number of symmetric segments to be read consecutively
    Npivot = Nsym - Ncon # number of symmetric segments to be read in a pivoting fashion
    Nasym = N - Nsym # number of asymmetric segments
    linear_start = Nasym + Nsym//2 - split_k0 - c # start of consecutively read segments
    linear_end = N - Npivot//2 # end of consecutively read segments (+1)
    linear = list(range(linear_start, linear_end)) # consecutively read segments
    if linear_start==Nasym:
        linear.reverse()
    # segments read in a pivoting fashion:
    pivot = [val for pair in zip(range(linear_end, N), reversed(range(Nasym, linear_start))) for val in pair]
    tail = list(range(min(linear_start, Nasym)))[::-1] # remaining asymmetric segments
    segment_order = linear + pivot + tail
    return segment_order


def place_waveform(waveform_floating, time):
    waveform = copy.deepcopy(waveform_floating)
    sequence.move_waveform(waveform, time)
    return waveform


def readtrain_shift(gr_echo_spacing, k0_gr_echo_index, num_gr_echoes):
    return gr_echo_spacing * (k0_gr_echo_index - (num_gr_echoes - 1) / 2)


def get_readtrain_spacing(TE, num_gr_echoes, gr_echo_spacing, k0_gr_echo_index, k0_rf_echo_index):
    readtrain_center = TE - readtrain_shift(gr_echo_spacing, k0_gr_echo_index, num_gr_echoes)
    return readtrain_center / (1 + k0_rf_echo_index)


def get_first_refocusing_time(is_gradient_echo, num_rf_echoes, TE, num_gr_echoes, gr_echo_spacing, k0_gr_echo_index, k0_rf_echo_index):
    if is_gradient_echo:
        return 0
    if (num_rf_echoes == 1):
        return TE / 2
    return get_readtrain_spacing(TE, num_gr_echoes, gr_echo_spacing, k0_gr_echo_index, k0_rf_echo_index) / 2


def get_RF_to_readtrain_center(TE, num_gr_echoes, gr_echo_spacing, k0_gr_echo_index, k0_rf_echo_index, first_refocusing_time):
    first_readtrain_center = get_readtrain_spacing(TE, num_gr_echoes, gr_echo_spacing, k0_gr_echo_index, k0_rf_echo_index)
    return first_readtrain_center - first_refocusing_time


def min_readtrain_spacing_from_k0_echo_indices(gr_echo_spacing, k0_gr_echo_index, num_gr_echoes, is_gradient_echo, min_RF_to_readtrain_center, num_rf_echoes, min_refocusing_time):
    TE_shift = readtrain_shift(gr_echo_spacing, k0_gr_echo_index, num_gr_echoes)
    if is_gradient_echo:
        return min_RF_to_readtrain_center + TE_shift
    min_RF_to_spin_echo = min_RF_to_readtrain_center
    if num_rf_echoes == 1:
        min_RF_to_spin_echo += TE_shift
    min_first_spin_echo = max(min_refocusing_time, min_RF_to_spin_echo) * 2
    min_first_readtrain_center = min_first_spin_echo
    if num_rf_echoes == 1:
        min_first_readtrain_center -= TE_shift
    return min_first_readtrain_center


def min_TE_from_k0_echo_indices(gr_echo_spacing, k0_gr_echo_index, num_gr_echoes, is_gradient_echo, min_RF_to_readtrain_center, num_rf_echoes, min_refocusing_time, k0_rf_echo_index):
    min_readtrain_spacing = min_readtrain_spacing_from_k0_echo_indices(gr_echo_spacing, k0_gr_echo_index, num_gr_echoes, is_gradient_echo, min_RF_to_readtrain_center, num_rf_echoes, min_refocusing_time)
    min_spin_echo_time = min_readtrain_spacing * (1 + k0_rf_echo_index)
    return min_spin_echo_time + readtrain_shift(gr_echo_spacing, k0_gr_echo_index, num_gr_echoes)


def get_k_coords(t, gp, tp, refocus_intervals):
    g = np.interp(t, tp, gp)
    dk = np.diff(t) * (g[:-1] + np.diff(g)/2) * constants.GYRO * 1e-3
    k = np.insert(np.cumsum(dk), 0, 0.) # start at k=0
    for (ref_start, ref_stop) in refocus_intervals:
        # k inversion of refocusing pulse corresponds to negative shift of 2k:
        k_before = k[t<=ref_start][-1]
        refocus_times = t[(t>ref_start) & (t<ref_stop)]
        k[(t>ref_start) & (t<ref_stop)] -= 2 * k_before * (refocus_times - ref_start) / (ref_stop - ref_start)
        k[t>=ref_stop] -= 2 * k_before
    return k


def bounds_hook(plot, elem, xbounds=None):
    x_range = plot.handles['plot'].x_range
    if xbounds is not None:
        x_range.bounds = xbounds
    else:
        x_range.bounds = x_range.start, x_range.end 


def hideframe_hook(plot, elem):
    plot.handles['plot'].outline_line_color = None


def flatten_dicts(list_of_dicts_and_lists):
    if list_of_dicts_and_lists is None:
        return []
    res = []
    for v in list_of_dicts_and_lists:
        res += flatten_dicts(v) if isinstance(v, list) else [v]
    return res


### Node functions ###


@Graph.node()
def phantom_object(object, min_voxel_size):
    return phantom.load(object, min_voxel_size)


@Graph.node()
def tissues(phantom_object):
    return list(phantom_object['shapes'].keys())


@Graph.node()
def is_radial(trajectory):
    return trajectory in ['Radial', 'PROPELLER']


@Graph.node()
def is_gradient_echo(sequence_type):
    return 'Gradient Echo' in sequence_type


@Graph.node()
def freq_dir(frequency_direction, is_radial):
    return constants.DIRECTIONS[frequency_direction] if not is_radial else 1


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
def rec_acq_ratio_F(recon_matrix_F, matrix_F):
    return recon_matrix_F / matrix_F


@Graph.node()
def rec_acq_ratio_P(recon_matrix_P, matrix_P):
    return recon_matrix_P / matrix_P


@Graph.node()
def min_TE(k0_echo_indices_linear_order, k0_echo_indices_reverse_linear_order, min_refocusing_time, min_RF_to_readtrain_center, gr_echo_spacing, EPI_factor, is_gradient_echo, turbo_factor):
    if EPI_factor == 1:
        gr_index, rf_index = 0, 0
        min_TE = min_TE_from_k0_echo_indices(gr_echo_spacing, gr_index, EPI_factor, is_gradient_echo, min_RF_to_readtrain_center, turbo_factor, min_refocusing_time, rf_index)
    else:
        min_TE_cands = []
        for indices in [k0_echo_indices_linear_order, k0_echo_indices_reverse_linear_order]:
            for (gr_index, rf_index) in indices:
                min_TE_cands.append(min_TE_from_k0_echo_indices(gr_echo_spacing, gr_index, EPI_factor, is_gradient_echo, min_RF_to_readtrain_center, turbo_factor, min_refocusing_time, rf_index))
        min_TE = min(min_TE_cands)
    return min([v for v in constants.PARAM_VALUES['TE'].values() if v >= min_TE])


@Graph.node()
def min_TR(spoiler, sequence_start):
    min_TR = spoiler['time'][-1] - sequence_start
    return min([v for v in constants.PARAM_VALUES['TR'].values() if v >= min_TR])


@Graph.node()
def max_TE(TR, sequence_start, spoiler_floating, readout_risetime, gre_echo_train_dur, EPI_factor, turbo_factor, k0_echo_indices_linear_order, k0_echo_indices_reverse_linear_order, gr_echo_spacing):
    last_readtrain_center = TR + sequence_start - spoiler_floating['dur_f'] + readout_risetime - gre_echo_train_dur / 2
    if EPI_factor == 1:
        max_TE = last_readtrain_center
    else:
        readtrain_spacing = last_readtrain_center / turbo_factor
        max_TE_cands = []
        for indices in [k0_echo_indices_linear_order, k0_echo_indices_reverse_linear_order]:
            for (gr_index, rf_index) in indices:
                readtrain_center = readtrain_spacing * (rf_index + 1)
                TE = readtrain_center + readtrain_shift(gr_echo_spacing, gr_index, EPI_factor)
                max_TE_cands.append(TE)
        max_TE = max(max_TE_cands)
    return max([v for v in constants.PARAM_VALUES['TE'].values() if v <= max_TE])


@Graph.node()
def max_readout_area(pixel_bandwidth, is_gradient_echo, k0_gr_echo_index, TE, RF_excitation, phaser_duration, slice_select_excitation, slice_select_rephaser, max_blip_dur, readtrain_spacing, refocusing_time, RF_refocusing_floating, EPI_factor):
    max_readout_areas = []
    # See paramBounds.tex for formulae
    d = 1e3 / pixel_bandwidth # readout duration
    s = constants.MAX_SLEW
    if is_gradient_echo:
        N = k0_gr_echo_index + 1/2
        M = k0_gr_echo_index * 2 + 1
        t = TE - RF_excitation['dur_f']/2
        v = 0 # gap between readouts
        for _ in range(2): # update readout gap after first pass
            if (M > 1):
                # max wrt G slice or G phase:
                q = t - max(phaser_duration,
                            slice_select_excitation['risetime_f'] + slice_select_rephaser['dur_f'])
                A = d*s*(q - N*(d+v) + v/2) / (M-1) # eq. 15
                max_readout_areas.append(A)
            # max wrt G read:
            h_roots = np.roots([8*(3-2*M), 4*s*(t*(2*M-6)+d*(2*N-M)+v*(2*N-1)), s**2*(4*t**2-d**2)]) # eq. 12
            h = min([h for h in h_roots if h>0] + [constants.MAX_AMP]) # truncate prephaser amp to max amp
            A_roots = np.roots([1, d*(d*s + 2*M*h), d**2*h*(2*h-s*(2*t-2*N*(d+v)+v))]) # eq. 13
            if np.all(A_roots<0):
                return 0 # no positive roots
            A = min([A for A in A_roots if A>0])
            max_readout_areas.append(A)
            read_risetime = min(max_readout_areas) / (d * s)
            v = max(max_blip_dur - 2 * read_risetime, 0)
    else: # (turbo) spin echo / GRASE
        # limit by half readout duration tr:
        tr = (readtrain_spacing - refocusing_time[0] - RF_refocusing_floating[0]['dur_f'] / 2) / EPI_factor
        Ar = d*s* tr - d**2*s/2
        max_readout_areas.append(Ar)
        # limit by prephaser duration tp:
        tp = refocusing_time[0] - RF_refocusing_floating[0]['dur_f'] / 2 - RF_excitation['dur_f'] / 2
        h = s * tp / 2
        h = min(h, constants.MAX_AMP)
        Ap = d * (np.sqrt((d*s)**2 - 8*h*(h-s*tp)) - d*s) / 2
        max_readout_areas.append(Ap)
    max_readout_areas.append(constants.MAX_AMP * 1e3 / pixel_bandwidth) # max wrt max_amp
    return min(max_readout_areas)


@Graph.node()
def max_phaser_area(is_gradient_echo, readtrain_spacing, RF_excitation, gre_echo_train_dur, readout_risetime, refocusing_time, RF_refocusing):
    if is_gradient_echo:
        max_phaser_duration = readtrain_spacing - RF_excitation['dur_f']/2 - gre_echo_train_dur/2 + readout_risetime
    else:
        max_phaser_duration = readtrain_spacing - refocusing_time[0] - RF_refocusing[0]['dur_f'] / 2 - gre_echo_train_dur / 2 + readout_risetime
    max_risetime = constants.MAX_AMP / constants.MAX_SLEW
    if max_phaser_duration > 2 * max_risetime: # trapezoid maxPhaser
        max_phaserarea = (max_phaser_duration - max_risetime) * constants.MAX_AMP
    else: # triangular maxPhaser
        max_phaserarea = (max_phaser_duration/2)**2 * constants.MAX_SLEW
    return max_phaserarea


@Graph.node()
def min_refocusing_time(is_gradient_echo, RF_excitation, RF_refocusing_floating, read_prephaser_floating, slice_select_excitation, slice_select_rephaser, slice_select_refocusing_floating):
    # Get earliest position of refocusing pulse
    if is_gradient_echo:
        return 0
    return RF_excitation['dur_f'] / 2 + RF_refocusing_floating[0]['dur_f'] / 2 + max(
        read_prephaser_floating['dur_f'], 
        slice_select_excitation['risetime_f'] + slice_select_rephaser['dur_f'] + (slice_select_refocusing_floating[0]['risetime_f'])
        )

@Graph.node()
def pixel_bandwidth_bounds(matrix_F, FOV_F, is_gradient_echo, RF_refocusing, turbo_factor, EPI_factor, TE, RF_excitation, refocusing_time, readtrain_spacing, TR, sequence_start, spoiler, k0_echo_indices_linear_order, k0_echo_indices_reverse_linear_order, reverse_linear_order, read_prephaser, phaser_duration, slice_select_excitation, slice_select_rephaser, max_blip_dur, slice_select_refocusing):
    readout_area = 1e3 * matrix_F / (FOV_F * constants.GYRO)
    # min limit imposed by maximum gradient amplitude:
    min_read_duration = readout_area / constants.MAX_AMP

    first_readtrain_ref = TE if (turbo_factor == 1) else readtrain_spacing
    last_readtrain_ref = TE if (turbo_factor == 1) else readtrain_spacing * turbo_factor
    last_read_end = TR + sequence_start - spoiler['dur_f']
    max_dur_ref_to_read_end = last_read_end - last_readtrain_ref
    
    slice_select_rewind_dur = slice_select_excitation['risetime_f'] + slice_select_rephaser['dur_f'] if is_gradient_echo else slice_select_refocusing[0]['risetime_f']

    k0_echo_indices = k0_echo_indices_linear_order if reverse_linear_order else k0_echo_indices_reverse_linear_order
    
    pixel_bandwidth_values = []
    for pixel_bandwidth in constants.PARAM_VALUES['pixel_bandwidth'].values():
        
        read_duration = 1e3 / pixel_bandwidth
        
        if read_duration < min_read_duration:
            continue
        
        readout_risetime = readout_area / read_duration / constants.MAX_SLEW
        readout_gap = max(max_blip_dur, 2 * readout_risetime)
        
        for (gr_echo, _) in k0_echo_indices:
            num_blips_before_ref = gr_echo if (turbo_factor == 1) else (EPI_factor-1) / 2
            num_blips_after_ref = EPI_factor - 1 - num_blips_before_ref
            
            dur_ref_to_read_end = read_duration * (num_blips_after_ref + 1/2) + readout_gap * num_blips_after_ref
            if dur_ref_to_read_end > max_dur_ref_to_read_end:
                continue
            
            if is_gradient_echo:
                first_read_start = RF_excitation['dur_f']/2 + max(
                    read_prephaser['dur_f'] + readout_risetime,
                    phaser_duration,
                    slice_select_rewind_dur
                    )
            else: # spin echo
                refocusing_dur = RF_refocusing[0]['dur_f']
                first_read_start = refocusing_time[0] + refocusing_dur / 2 + max(
                    readout_risetime,
                    phaser_duration,
                    slice_select_rewind_dur
                    )
            
            max_dur_read_start_to_ref = first_readtrain_ref - first_read_start
            dur_read_start_to_ref = read_duration * (num_blips_before_ref + 1/2) + readout_gap * num_blips_before_ref
            
            if dur_read_start_to_ref > max_dur_read_start_to_ref:
                continue

            pixel_bandwidth_values.append(pixel_bandwidth)

    return MinMax(min(pixel_bandwidth_values, default=np.inf), max(pixel_bandwidth_values, default=-np.inf))

@Graph.node()
def matrix_F_bounds(max_readout_area, FOV_F, parameter_style, FOV_bandwidth, pixel_bandwidth_bounds):
    min_matrix_F = []
    max_matrix_F = [max_readout_area * 1e-3 * FOV_F * constants.GYRO]
    if parameter_style == 'Matrix and FOV BW': # constant FOV BW puts contraints on matrix_F
        min_matrix_F.append(FOV_bandwidth * 2e3 / pixel_bandwidth_bounds.max)
        max_matrix_F.append(FOV_bandwidth * 2e3 / pixel_bandwidth_bounds.min)
    return MinMax(max(min_matrix_F, default=-np.inf), min(max_matrix_F))

@Graph.node()
def matrix_P_bounds(max_phaser_area, FOV_P):
    max_matrix_P = int(max_phaser_area * 2e-3 * FOV_P * constants.GYRO) + 1
    return MinMax(-np.inf, max_matrix_P)

@Graph.node()
def recon_matrix_F_bounds(matrix_F):
    return MinMax(matrix_F, np.inf)

@Graph.node()
def recon_matrix_P_bounds(matrix_P):
    return MinMax(matrix_P, np.inf)

@Graph.node()
def FOV_F_bounds(matrix_F, max_readout_area, parameter_style, voxel_F, matrix_F_bounds, recon_voxel_F, recon_matrix_F_bounds):
    min_FOV_F = [1e3 * matrix_F / (max_readout_area * constants.GYRO) if max_readout_area > 0 else np.inf]
    max_FOV_F = []
    if parameter_style == 'Voxel size and Fat/water shift': # constant voxel size puts constraints on FOV
        min_FOV_F.append(voxel_F * matrix_F_bounds.min)
        min_FOV_F.append(recon_voxel_F * recon_matrix_F_bounds.min)
        max_FOV_F.append(voxel_F * matrix_F_bounds.max)
        max_FOV_F.append(recon_voxel_F * recon_matrix_F_bounds.max)
    return MinMax(max(min_FOV_F), min(max_FOV_F, default=np.inf))

@Graph.node()
def FOV_P_bounds(matrix_P, max_phaser_area, parameter_style, matrix_P_bounds, voxel_P, recon_voxel_P, recon_matrix_P_bounds):
    min_FOV_P = [(matrix_P - 1) / (max_phaser_area * constants.GYRO * 2e-3)]
    max_FOV_P = []
    if parameter_style == 'Voxel size and Fat/water shift': # constant voxel size puts constraints on FOV
        min_FOV_P.append(voxel_P * matrix_P_bounds.min)
        min_FOV_P.append(recon_voxel_P * recon_matrix_P_bounds.min)
        max_FOV_P.append(voxel_P * matrix_P_bounds.max)
        max_FOV_P.append(recon_voxel_P * recon_matrix_P_bounds.max)
    return MinMax(max(min_FOV_P), min(max_FOV_P, default=np.inf))

@Graph.node()
def min_RF_to_readtrain_center(is_gradient_echo, RF_excitation, read_prephaser_floating, slice_select_excitation, slice_select_rephaser, RF_refocusing_floating, slice_select_refocusing_floating, gre_echo_train_dur, readout_risetime, phaser_duration):
    # Get shortest spacing bewteen RF (excitation or refocusing for gradient / spin echo) and center of readout (train)
    if is_gradient_echo:
        RF_dur = RF_excitation['dur_f']
        read_prephaser_dur = read_prephaser_floating['dur_f']
        slice_rewind_dur = slice_select_excitation['risetime_f'] + slice_select_rephaser['dur_f']
    else: # spin echo
        RF_dur = RF_refocusing_floating[0]['dur_f']
        read_prephaser_dur = 0
        slice_rewind_dur = slice_select_refocusing_floating[0]['risetime_f']
    return RF_dur / 2 + gre_echo_train_dur / 2 - readout_risetime + max(
        read_prephaser_dur + readout_risetime,
        phaser_duration,
        slice_rewind_dur
        )


@Graph.node()
def k0_echo_indices_linear_order(k0_segment, turbo_factor):
    # (k0_gr_echo_index, k0_rf_echo_index)
    return [(segment // turbo_factor, segment % turbo_factor) for segment in k0_segment]


@Graph.node()
def k0_echo_indices_reverse_linear_order(k0_echo_indices_linear_order, EPI_factor, turbo_factor):
    return {(EPI_factor - 1 - gr_index, turbo_factor - 1 - rf_index) for (gr_index, rf_index) in k0_echo_indices_linear_order}


@Graph.node()
def k0_index(k0_echo_indices_reverse_linear_order, k0_echo_indices_linear_order, is_gradient_echo, turbo_factor, TE, EPI_factor, gr_echo_spacing, min_refocusing_time, min_RF_to_readtrain_center):
    if EPI_factor == 1:
        gr_index = 0
        reverse_order = False
        if is_gradient_echo or turbo_factor == 1:
            rf_index = 0
        else: # flexible segment order
            min_readtrain_spacing = min_readtrain_spacing_from_k0_echo_indices(gr_echo_spacing, gr_index, EPI_factor, is_gradient_echo, min_RF_to_readtrain_center, turbo_factor, min_refocusing_time)
            rf_index = int(np.floor(TE / min_readtrain_spacing)) - 1
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
def readtrain_spacing(EPI_factor, gr_echo_spacing, TE, k0_gr_echo_index, k0_rf_echo_index):
    # Equals center position of gradient echo (train) for gradient echo sequences
    # Equals rf echo spacing for spin echo sequences
    return get_readtrain_spacing(TE, EPI_factor, gr_echo_spacing, k0_gr_echo_index, k0_rf_echo_index)


@Graph.node()
def num_blades(is_radial, matrix, radial_factor, turbo_factor, EPI_factor):
    return int(np.ceil(max(matrix) * radial_factor / turbo_factor / EPI_factor * np.pi / 2)) if is_radial else 1


@Graph.node()
def k_angles(num_blades):
    return np.linspace(0, np.pi, num_blades, endpoint=False)


@Graph.node()
def spoke_angle(k_angles, shot):
    return np.degrees(k_angles[min(shot-1, len(k_angles)-1)])


@Graph.node()
def num_shots(matrix_P, phase_oversampling, partial_Fourier, turbo_factor, EPI_factor, is_radial, num_blades):
    return int(np.ceil(matrix_P * (1 + phase_oversampling / 100) * partial_Fourier / turbo_factor / EPI_factor)) if not is_radial else num_blades


@Graph.node()
def shot_label(is_radial, EPI_factor, turbo_factor):
    return 'shot' if not is_radial else 'spoke' if (EPI_factor * turbo_factor == 1) else 'blade'


@Graph.node()
def num_measured_lines(turbo_factor, EPI_factor, num_shots, is_radial):
    # measured lines per blade
    return turbo_factor * EPI_factor * (num_shots if not is_radial else 1)


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
def k_phase_axis(is_radial, num_measured_lines, matrix, phase_dir, phase_oversampling, FOV):
    if not is_radial:
        # oversampling may be higher than prescribed since num_shots must be integer:
        num_lines = max(num_measured_lines, int(np.ceil(matrix[phase_dir] * (1 + phase_oversampling / 100))))
        voxel_size = FOV[phase_dir] / matrix[phase_dir]
    else:
        num_lines = num_measured_lines # future: take undersampling into account
        voxel_size = max(FOV) / num_lines # corresponding to blade width
    return recon.get_k_axis(num_lines, voxel_size)


@Graph.node()
def num_blank_lines(k_phase_axis, lines_to_measure):
    return len(k_phase_axis) - sum(lines_to_measure)


@Graph.node()
def lines_to_measure(k_phase_axis, num_measured_lines):
    lines_to_measure = np.ones(len(k_phase_axis), dtype=bool)
    # undersample by partial Fourier:
    lines_to_measure[num_measured_lines:] = False
    assert(sum(lines_to_measure) == num_measured_lines)
    return lines_to_measure


@Graph.node()
def num_segm(turbo_factor, EPI_factor):
    # number of k-space segments
    return turbo_factor * EPI_factor


@Graph.node()
def num_sym_lines(num_measured_lines, k_phase_axis):
    return 2 * num_measured_lines - len(k_phase_axis)


@Graph.node()
def num_sym_segm(num_segm, num_sym_lines, num_measured_lines):
    # number of k-space segments symmetric about k0:
    num_sym_segm = num_segm * (num_sym_lines / num_measured_lines)
    if (num_sym_segm % 2 == 0):
        return int(num_sym_segm) # k0 lies between two segments
    return int(np.round((num_sym_segm - 1) / 2)) * 2 + 1


@Graph.node()
def k0_segment(num_segm, num_sym_segm):
    k0_segment = num_segm - num_sym_segm // 2 - 1
    if (num_sym_segm % 2 == 0):
        return [k0_segment, k0_segment + 1] # k0 lies between two segments
    return [k0_segment]


@Graph.node()
def pe_table(num_measured_lines, num_segm, num_shots, EPI_factor, turbo_factor, num_sym_segm, k0_rf_echo_index, reverse_linear_order):
    num_lines_per_segment = int(num_measured_lines / num_segm)
    lines = [shot % num_lines_per_segment for shot in range(num_shots)]
    if EPI_factor == 1: # (turbo) spin echo
        segment_order = np.array(get_segment_order(turbo_factor, num_sym_segm, k0_rf_echo_index)).reshape(-1, 1)
    else: # EPI and GRASE
        order = -1 if reverse_linear_order else 1
        segment_order = np.array(range(EPI_factor))[None, ::order] * turbo_factor + np.array(range(turbo_factor))[::order, None]
    return num_lines_per_segment * segment_order + np.array(lines)[:, None, None]


@Graph.node()
def k_axes(freq_dir, phase_dir, k_read_axis, k_phase_axis, lines_to_measure):
    k_axes = [None]*2
    k_axes[freq_dir] = k_read_axis
    k_axes[phase_dir] = k_phase_axis[lines_to_measure]
    return k_axes


@Graph.node()
def k_samples(k_axes, k_angles):
    k_samples = np.array(np.meshgrid(k_axes[0], k_axes[1])).T
    # rotate samples for each angle:
    rotmat = np.array([[np.cos(k_angles), -np.sin(k_angles)], 
                        [np.sin(k_angles),  np.cos(k_angles)]])
    return np.einsum('ijk,klm->ijml', k_samples, rotmat) # shape=(Nx, Ny, Nangles, 2)


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
def plain_kspace_comps(is_radial, phantom_object, k_grid_axes, k_samples):
    if not is_radial:
        return recon.resample_kspace_Cartesian(phantom_object, k_grid_axes, shape=k_samples.shape[:-1])
    return recon.resample_kspace(phantom_object, k_samples)


@Graph.node()
def thick_kspace_comps(slice_thickness, k_samples, plain_kspace_comps):
    # Lorenzian line shape to mimic slice thickness
    blur_factor = .5
    slice_thickness_filter = slice_thickness * np.exp(-blur_factor * slice_thickness * np.sqrt(np.sum(k_samples**2, axis=-1)))
    thick_kspace_comps = {}
    for tissue in plain_kspace_comps:
        thick_kspace_comps[tissue] = plain_kspace_comps[tissue] * slice_thickness_filter
    return thick_kspace_comps


@Graph.node()
def signal_level(k_read_axis, lines_to_measure, num_blades, slice_thickness, FOV, matrix):
    return np.sqrt(len(k_read_axis) * sum(lines_to_measure) * num_blades) * slice_thickness * np.prod(FOV) / np.prod(matrix)


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
def time_relative_inphase(time_after_excitation, is_gradient_echo, spin_echoes, phase_dir):
    time_relative_inphase = copy.deepcopy(time_after_excitation)
    if not is_gradient_echo:
        # for spinecho, subtract Hahn echo position from time_after_excitation
        time_relative_inphase -= np.expand_dims(spin_echoes, axis=[dim for dim in range(3) if dim != phase_dir])
    return time_relative_inphase


@Graph.node()
def dephasing(field_strength, time_relative_inphase):
    dephasing = {}
    for component, resonance in constants.FAT_RESONANCES.items():
        dephasing[component] = np.exp(2j*np.pi * constants.GYRO * field_strength * resonance['shift'] * time_relative_inphase * 1e-3)
    return dephasing


@Graph.node()
def T2w(tissues, time_after_excitation, time_relative_inphase, field_strength):
    T2w = {}
    for tissue in tissues:
        T2w[tissue] = get_T2w(tissue, time_after_excitation, time_relative_inphase, field_strength)
        if constants.TISSUES[tissue]['FF'] > 0: # fat containing tissues
            for component in constants.FAT_RESONANCES:
                T2w[tissue + component] = get_T2w(component, time_after_excitation, time_relative_inphase, field_strength)
    return T2w


@Graph.node()
def kspace_comps(tissues, thick_kspace_comps, T2w, dephasing):
    kspace_comps = {}
    for tissue in tissues:
        if constants.TISSUES[tissue]['FF'] == .00:
            kspace_comps[tissue] = thick_kspace_comps[tissue] * T2w[tissue]
        else: # fat containing tissues
            kspace_comps[tissue + 'Water'] = thick_kspace_comps[tissue] * T2w[tissue]
            for component in constants.FAT_RESONANCES:
                kspace_comps[tissue + component] = thick_kspace_comps[tissue] * dephasing[component] * T2w[tissue + component]
    return kspace_comps


@Graph.node()
def decayed_signal(signal_level, T2w, reference_tissue, k_read_axis, k_phase_axis, freq_dir):
    return signal_level * np.take(np.take(T2w[reference_tissue], np.argmin(np.abs(k_read_axis)), axis=freq_dir), np.argmin(np.abs(k_phase_axis)))


@Graph.node()
def noise(k_samples, noise_std):
    sampled_matrix = k_samples.shape[:-1]
    return np.random.normal(0, noise_std, sampled_matrix) + 1j * np.random.normal(0, noise_std, sampled_matrix)


@Graph.node()
def SNR(reference_signal, noise_std):
    return reference_signal / noise_std


@Graph.node()
def relative_SNR(SNR, reference_SNR):
    return SNR / reference_SNR * 100


@Graph.node()
def PD_and_T1w(sequence_type, TR, TI, FA, field_strength, tissues):
    return {component: get_PD_and_T1w(component, sequence_type, TR, TI, FA, field_strength) for component in set(tissues).union(set(constants.FAT_RESONANCES.keys()))}


@Graph.node()
def reference_signal(decayed_signal, PD_and_T1w, reference_tissue):
    return decayed_signal * np.abs(PD_and_T1w[reference_tissue])


@Graph.node()
def scantime(num_shots, NSA, TR):
    return formatting.format_scantime(num_shots * NSA * TR)


@Graph.node()
def measured_kspace(noise, kspace_comps, FatSat, PD_and_T1w):
    measured_kspace = copy.deepcopy(noise)
    for component in kspace_comps:
        if 'Fat' in component:
            tissue = component[:component.find('Fat')]
            resonance = component[component.find('Fat'):]
            ratio = constants.FAT_RESONANCES[resonance]['ratio_with_FatSat' if FatSat else 'ratio']
            ratio *= constants.TISSUES[tissue]['FF']
            measured_kspace += kspace_comps[component] * PD_and_T1w[resonance] * ratio
        else:
            if 'Water' in component:
                tissue = component[:component.find('Water')]
                ratio = 1 - constants.TISSUES[tissue]['FF']    
            else:
                tissue = component
                ratio = 1.0
            measured_kspace += kspace_comps[component] * PD_and_T1w[tissue] * ratio
    return measured_kspace


@Graph.node()
def gridded_kspace(k_grid_axes, is_radial, measured_kspace, k_samples, FOV, matrix):
    grid_shape = tuple(len(k_grid_axes[dim]) for dim in range(2))
    if not is_radial:
        return measured_kspace.reshape(grid_shape)
    samples = k_samples * FOV / matrix
    return recon.grid(measured_kspace, grid_shape, samples)


@Graph.node()
def full_kspace(num_blank_lines, is_radial, gridded_kspace, phase_dir, homodyne, k_phase_axis):
    if (num_blank_lines == 0 or is_radial):
        return np.copy(gridded_kspace)
    shape_unsampled = tuple(num_blank_lines if dim==phase_dir else n for dim, n in enumerate(gridded_kspace.shape))
    full_kspace = np.append(gridded_kspace, np.zeros(shape_unsampled), axis=phase_dir) # zerofill
    if homodyne and (num_blank_lines > 0):
        full_kspace *= recon.homodyne_weights(len(k_phase_axis), num_blank_lines, phase_dir) # pre-weighting
        full_kspace += np.conjugate(np.flip(full_kspace))
    return full_kspace


@Graph.node()
def full_k_matrix(full_kspace):
    return full_kspace.shape


@Graph.node()
def apodized_kspace(full_kspace, do_apodize, apodization_alpha):
    apodized_kspace = copy.deepcopy(full_kspace)
    if do_apodize: 
        apodized_kspace *= recon.radial_Tukey(apodization_alpha, full_kspace.shape)
    return apodized_kspace


@Graph.node()
def oversampled_recon_matrix(recon_matrix, full_k_matrix, matrix):
    oversampled_recon_matrix = copy.deepcopy(recon_matrix)
    for dim in range(2):
        oversampled_recon_matrix[dim] = int(np.round(recon_matrix[dim] * full_k_matrix[dim] / matrix[dim]))
    return oversampled_recon_matrix


@Graph.node()
def zerofilled_kspace(apodized_kspace, oversampled_recon_matrix):
    return recon.zerofill(apodized_kspace, oversampled_recon_matrix)


@Graph.node()
def image_array(oversampled_recon_matrix, full_k_matrix, recon_matrix, zerofilled_kspace):
    pixel_shifts = [0., 0.]
    sample_shifts = [0., 0.]
    for dim in range(2):
        if not oversampled_recon_matrix[dim]%2:
            pixel_shifts[dim] += 1/2 # half pixel shift for even matrixsize due to fft
        if (oversampled_recon_matrix[dim] - recon_matrix[dim])%2:
            pixel_shifts[dim] += 1/2 # half pixel shift due to cropping an odd number of pixels in image space
        if not full_k_matrix[dim]%2:
            sample_shifts[dim] += 1/2 # half sample shift for even matrixsize due to fft
            if (oversampled_recon_matrix[dim] - full_k_matrix[dim])%2:
                sample_shifts[dim] -= 1 # sample shift for odd number of zeroes added
    image_array = recon.IFFT(zerofilled_kspace, pixel_shifts, sample_shifts)
    return recon.crop(image_array, recon_matrix)


@Graph.node()
def RF_excitation(FA, is_gradient_echo):
    flip_angle = FA if is_gradient_echo else 90.
    return sequence.get_RF(flip_angle=flip_angle, time=0., dur=3., shape='hamming_sinc',  name='excitation')


@Graph.node()
def RF_refocusing_floating(is_gradient_echo, turbo_factor):
    if is_gradient_echo:
        return None
    RF_refocusing = []
    for rf_echo in range(turbo_factor):
        RF_refocusing.append(sequence.get_RF(flip_angle=180., dur=3., shape='hamming_sinc',  name=f'refocusing{" " + str(rf_echo + 1) if turbo_factor > 1 else ""}'))
    return RF_refocusing


@Graph.node()
def RF_inversion_floating(sequence_type):
    if not sequence_type == 'Inversion Recovery':
        return None
    return sequence.get_RF(flip_angle=180., dur=3., shape='hamming_sinc',  name='inversion')


@Graph.node()
def RF_FatSat_floating(FatSat, field_strength):
    if not FatSat:
        return None
    return sequence.get_RF(flip_angle=90, time=0., dur=30./field_strength, shape='hamming_sinc',  name='FatSat')


@Graph.node()
def FatSat_spoiler_floating(FatSat):
    if not FatSat:
        return None
    spoiler_area = 30. # uTs/m
    return sequence.get_gradient('slice', total_area=spoiler_area, name='FatSat spoiler', max_amp=constants.MAX_AMP, max_slew=constants.MAX_SLEW)


@Graph.node()
def slice_select_excitation(RF_excitation, slice_thickness):
    flat_dur = RF_excitation['dur_f']
    amp = RF_excitation['FWHM_f'] / (slice_thickness * constants.GYRO)
    time = 0.
    return sequence.get_gradient('slice', time, max_amp=amp, flat_dur=flat_dur, name='slice select excitation', max_slew=constants.MAX_SLEW)


@Graph.node()
def slice_select_rephaser(slice_select_excitation):
    slice_rephaser_area = -slice_select_excitation['area_f']/2
    slice_select_rephaser = sequence.get_gradient('slice', total_area=slice_rephaser_area, name='slice select rephaser', max_amp=constants.MAX_AMP, max_slew=constants.MAX_SLEW)
    time = (slice_select_excitation['dur_f'] + slice_select_rephaser['dur_f']) / 2
    sequence.move_waveform(slice_select_rephaser, time)
    return slice_select_rephaser


@Graph.node()
def slice_select_refocusing_floating(RF_refocusing_floating, slice_thickness, turbo_factor):
    if RF_refocusing_floating is None:
        return None
    flat_dur = RF_refocusing_floating[0]['dur_f']
    amp = RF_refocusing_floating[0]['FWHM_f'] / (slice_thickness * constants.GYRO)
    slice_select_refocusing = []
    for rf_echo in range(turbo_factor):
        slice_select_refocusing.append(sequence.get_gradient('slice', max_amp=amp, flat_dur=flat_dur, name='slice select refocusing', max_slew=constants.MAX_SLEW))
    return slice_select_refocusing


@Graph.node()
def slice_select_inversion_floating(sequence_type, RF_inversion_floating, slice_thickness):
    if sequence_type != 'Inversion Recovery':
        return None
    flat_dur = RF_inversion_floating['dur_f']
    amp = RF_inversion_floating['FWHM_f'] / (constants.INVERSION_THK_FACTOR * slice_thickness * constants.GYRO)
    return sequence.get_gradient('slice', max_amp=amp, flat_dur=flat_dur, name='slice select inversion', max_slew=constants.MAX_SLEW)


@Graph.node()
def inversion_spoiler_floating(sequence_type):
    if sequence_type != 'Inversion Recovery':
        return None
    spoiler_area = 30. # uTs/m
    return sequence.get_gradient('slice', total_area=spoiler_area, name='inversion spoiler', max_amp=constants.MAX_AMP, max_slew=constants.MAX_SLEW)


@Graph.node()
def readouts_floating(k_read_axis, pixel_bandwidth, matrix_F, FOV_F, turbo_factor, EPI_factor):
    pixel_size = (len(k_read_axis)-1) / len(k_read_axis) / (max(k_read_axis)-min(k_read_axis))
    flat_area = 1e3 / pixel_size / constants.GYRO # uTs/m
    amp = pixel_bandwidth * matrix_F / (FOV_F * constants.GYRO) # mT/m
    readouts = []
    for rf_echo in range(turbo_factor):
        readouts.append([])           
        for gr_echo in range(EPI_factor):
            suffix = ((" " if (turbo_factor > 1 or EPI_factor > 1) else "")
                    + (str(rf_echo + 1) if turbo_factor > 1 else "")
                    + ("." if (turbo_factor > 1 and EPI_factor > 1) else "")
                    + (str(gr_echo + 1) if EPI_factor > 1 else ""))
            readout = sequence.get_gradient('frequency', max_amp=amp, flat_area=flat_area, name='readout'+suffix, max_slew=constants.MAX_SLEW)
            if gr_echo % 2: # even EPI echoes must have negative polarity
                sequence.rescale_gradient(readout, -1)
            readouts[-1].append(readout)
    return readouts


@Graph.node()
def sampling_windows_floating(turbo_factor, EPI_factor, readouts_floating):
    sampling_windows = []
    for rf_echo in range(turbo_factor):
        sampling_windows.append([])
        for gr_echo in range(EPI_factor):
            suffix = ((" " if (turbo_factor > 1 or EPI_factor > 1) else "")
                    + (str(rf_echo + 1) if turbo_factor > 1 else "")
                    + ("." if (turbo_factor > 1 and EPI_factor > 1) else "")
                    + (str(gr_echo + 1) if EPI_factor > 1 else ""))
            adc = sequence.get_ADC(dur=readouts_floating[0][0]['flat_dur_f'], name='sampling'+suffix)
            sampling_windows[-1].append(adc)
    return sampling_windows


@Graph.node()
def readout_risetime(readouts_floating):
    return readouts_floating[0][0]['risetime_f']


@Graph.node()
def read_prephaser_floating(readouts_floating, is_gradient_echo):
    read_prephaser = sequence.get_gradient('frequency', total_area=readouts_floating[0][0]['area_f']/2, name='read prephaser', max_amp=constants.MAX_AMP, max_slew=constants.MAX_SLEW)
    if is_gradient_echo:
        sequence.rescale_gradient(read_prephaser, -1)
    return read_prephaser


@Graph.node()
def phase_step_area(k_phase_axis):
    return np.mean(np.diff(k_phase_axis)) * 1e3 / constants.GYRO # uTs/m


@Graph.node()
def largest_phaser_area(k_phase_axis):
    return np.min(k_phase_axis) * 1e3 / constants.GYRO # uTs/m


@Graph.node()
def phaser_duration(largest_phaser_area):
    largest_phaser = sequence.get_gradient('phase', total_area=largest_phaser_area, max_amp=constants.MAX_AMP, max_slew=constants.MAX_SLEW)
    return largest_phaser['dur_f']


@Graph.node()
def max_blip_dur(EPI_factor, phase_step_area, num_shots, turbo_factor):
    if (EPI_factor <= 1):
        return 0
    max_blip_area = phase_step_area * num_shots * turbo_factor
    max_blip = sequence.get_gradient('phase', total_area=max_blip_area, max_amp=constants.MAX_AMP, max_slew=constants.MAX_SLEW)
    return max_blip['dur_f']


@Graph.node()
def readout_gap(max_blip_dur, readouts_floating):
    return max(max_blip_dur - 2 * readouts_floating[0][0]['risetime_f'], 0)


@Graph.node()
def gr_echo_spacing(readouts_floating, readout_gap):
    return readouts_floating[0][0]['dur_f'] + readout_gap


@Graph.node()
def gre_echo_train_dur(EPI_factor, gr_echo_spacing, readout_gap):
    return EPI_factor * gr_echo_spacing - readout_gap


@Graph.node()
def phasers_floating(turbo_factor, largest_phaser_area, pe_table, phase_step_area, shot):
    phasers = []
    for rf_echo in range(turbo_factor):
        phaser_area = largest_phaser_area + pe_table[shot-1, rf_echo, 0] * phase_step_area
        suffix = f' {rf_echo + 1}' if turbo_factor > 1 else ''
        phaser = sequence.get_gradient('phase', total_area=largest_phaser_area, name='phase encode'+suffix, max_amp=constants.MAX_AMP, max_slew=constants.MAX_SLEW)
        if abs(largest_phaser_area) > 1e-5:
            sequence.rescale_gradient(phaser, phaser_area / largest_phaser_area)
        phasers.append(phaser)
    return phasers


@Graph.node()
def blips_floating(turbo_factor, EPI_factor, phase_step_area, pe_table, shot):
    blips = []
    for rf_echo in range(turbo_factor):
        blips.append([])
        for gr_echo in range(1, EPI_factor):
            blip_area = phase_step_area * (pe_table[shot-1, rf_echo, gr_echo] - pe_table[shot-1, rf_echo, gr_echo-1])
            blip = sequence.get_gradient('phase', total_area=blip_area, name='blip', max_amp=constants.MAX_AMP, max_slew=constants.MAX_SLEW)
            blips[-1].append(blip)
    return blips


@Graph.node()
def rephasers_floating(turbo_factor, phasers_floating, blips_floating, largest_phaser_area):
    rephasers = []
    for rf_echo in range(turbo_factor):
        suffix = f' {rf_echo + 1}' if turbo_factor > 1 else ''
        rephaser_area = -phasers_floating[rf_echo]['area_f']
        for blip in blips_floating[rf_echo]:
            rephaser_area -= blip['area_f']
        rephaser = sequence.get_gradient('phase', total_area=largest_phaser_area, name='rephaser'+suffix, max_amp=constants.MAX_AMP, max_slew=constants.MAX_SLEW)
        if abs(largest_phaser_area) > 1e-5:
            sequence.rescale_gradient(rephaser, rephaser_area / largest_phaser_area)
        rephasers.append(rephaser)
    return rephasers


@Graph.node()
def spoiler_floating():
    spoiler_area = 30. # uTs/m
    return sequence.get_gradient('slice', total_area=spoiler_area, name='spoiler', max_amp=constants.MAX_AMP, max_slew=constants.MAX_SLEW)


@Graph.node()
def readtrain_center_time(readtrain_spacing, turbo_factor):
    # center position of gradient echo readout (train)(s)
    return [readtrain_spacing * (rf_echo + 1) for rf_echo in range(turbo_factor)]


@Graph.node()
def readout_center_time(EPI_factor, gr_echo_spacing, readtrain_center_time):
    return [[center_time + (gre - (EPI_factor-1) / 2) * gr_echo_spacing for gre in range(EPI_factor)] for center_time in readtrain_center_time]


@Graph.node()
def refocusing_time(TE, readtrain_center_time, readtrain_spacing):
    if len(readtrain_center_time) == 1:
        return [TE / 2]
    return [t - readtrain_spacing / 2 for t in readtrain_center_time]


@Graph.node()
def slice_select_refocusing(slice_select_refocusing_floating, refocusing_time):
    if slice_select_refocusing_floating is None:
        return None
    return [place_waveform(grad, time) for grad, time in zip(slice_select_refocusing_floating, refocusing_time)]


@Graph.node()
def RF_refocusing(RF_refocusing_floating, refocusing_time):
    if RF_refocusing_floating is None:
        return None
    return [place_waveform(RF, time) for RF, time in zip(RF_refocusing_floating, refocusing_time)]


@Graph.node()
def slice_select_inversion(slice_select_inversion_floating, TI):
    if slice_select_inversion_floating is None:
        return None
    return place_waveform(slice_select_inversion_floating, -TI)


@Graph.node()
def RF_inversion(RF_inversion_floating, TI):
    if RF_inversion_floating is None:
        return None
    return place_waveform(RF_inversion_floating, -TI)


@Graph.node()
def inversion_spoiler(inversion_spoiler_floating, RF_inversion):
    if inversion_spoiler_floating is None:
        return None
    time = RF_inversion['time'][-1] + inversion_spoiler_floating['dur_f']/2
    return place_waveform(inversion_spoiler_floating, time)


@Graph.node()
def FatSat_spoiler(FatSat_spoiler_floating, slice_select_excitation):
    if FatSat_spoiler_floating is None:
        return None
    time = slice_select_excitation['time'][0] - FatSat_spoiler_floating['dur_f']/2
    return place_waveform(FatSat_spoiler_floating, time)


@Graph.node()
def RF_FatSat(RF_FatSat_floating, FatSat_spoiler_floating):
    if RF_FatSat_floating is None:
        return None
    time = FatSat_spoiler_floating['time'][0] - RF_FatSat_floating['dur_f']/2
    return place_waveform(RF_FatSat_floating, time)


@Graph.node()
def readouts(readouts_floating, readout_center_time):
    return [[place_waveform(readout, time) for readout, time in zip(readouts, times)] for readouts, times in zip(readouts_floating, readout_center_time)]


@Graph.node()
def sampling_windows(sampling_windows_floating, readout_center_time):
    return [[place_waveform(sampling, time) for sampling, time in zip(samplings, times)] for samplings, times in zip(sampling_windows_floating, readout_center_time)]


@Graph.node()
def read_prephaser(read_prephaser_floating, is_gradient_echo, readouts, RF_excitation):
    if is_gradient_echo:
        first_readout = readouts[0][0]
        time = first_readout['center_f'] - (read_prephaser_floating['dur_f'] + first_readout['dur_f']) / 2
    else:
        time = (RF_excitation['dur_f'] + read_prephaser_floating['dur_f']) / 2
    return place_waveform(read_prephaser_floating, time)


@Graph.node()
def phasers(readtrain_center_time, phasers_floating, gre_echo_train_dur, readout_risetime):
    return [place_waveform(phaser, center - (gre_echo_train_dur + phaser['dur_f'])/2 + readout_risetime) for phaser, center in zip(phasers_floating, readtrain_center_time)]


@Graph.node()
def rephasers(readtrain_center_time, gre_echo_train_dur, readout_risetime, rephasers_floating):
    return [place_waveform(rephaser, center + (gre_echo_train_dur + rephaser['dur_f'])/2 - readout_risetime) for rephaser, center in zip(rephasers_floating, readtrain_center_time)]


@Graph.node()
def blips(readtrain_center_time, EPI_factor, gr_echo_spacing, blips_floating):
    return [[place_waveform(blip, center + gr_echo_spacing * (gre - EPI_factor/2 + 1)) for gre, blip in enumerate(blips)] for center, blips in zip(readtrain_center_time, blips_floating)]


@Graph.node()
def spoiler(readouts, spoiler_floating):
    time = readouts[-1][-1]['center_f'] + (readouts[-1][-1]['flat_dur_f'] + spoiler_floating['dur_f']) / 2
    return place_waveform(spoiler_floating, time)


@Graph.node()
def sequence_start(slice_select_inversion, RF_FatSat, slice_select_excitation):
    if slice_select_inversion is not None:
        return slice_select_inversion['time'][0]
    elif RF_FatSat is not None:
        return RF_FatSat['time'][0]
    else:
        return slice_select_excitation['time'][0]


@Graph.node()
def signal_curves(measured_kspace, shot, is_radial, turbo_factor, EPI_factor, pe_table, phase_dir, time_after_excitation):
    signal_curves = []
    scale = 1 / np.max(np.abs(np.real(measured_kspace)))
    signal_exponent = .5
    spoke = shot-1 if is_radial else 0
    for rf_echo in range(turbo_factor):
        signal_curves.append([])
        for gr_echo in range(EPI_factor):
            ky = pe_table[shot-1, rf_echo, gr_echo]
            waveform = np.real(np.take(measured_kspace[..., spoke], indices=ky, axis=phase_dir))
            t = np.take(time_after_excitation[..., spoke if spoke<time_after_excitation.shape[-1] else 0], indices=ky, axis=phase_dir)
            signal = sequence.get_signal(waveform, t, scale, signal_exponent)
            signal_curves[-1].append(signal)
    return signal_curves


@Graph.node()
def k_trajectory(RF_refocusing, frequency_board, phase_board, is_radial, phase_dir, spoke_angle):
    frequency_area = frequency_board['net_gradient']
    phase_area = phase_board['net_gradient']
    dt = .01
    refocus_intervals = [list(rf['time'][[0, -1]]) for rf in RF_refocusing] if RF_refocusing else []
    t = np.concatenate((*(area['time'] for area in [frequency_area, phase_area]), [t for ref in refocus_intervals for t in ref])) # k event times
    t = np.unique(np.concatenate((t, np.arange(0., max(t), dt)))) # merge with time grid
    kx = get_k_coords(t, *(frequency_area[dim] for dim in ['G read', 'time']), refocus_intervals)
    ky = get_k_coords(t, *(phase_area[dim] for dim in ['G phase', 'time']), refocus_intervals)
    if not is_radial:
        if phase_dir==1:
            kx, ky = ky, kx
    else: # rotate by spoke/blade angle
        angle = np.radians(spoke_angle)
        cos, sin = np.cos(angle), np.sin(angle)
        kx, ky = cos * kx - sin * ky, sin * kx + cos * ky
    return {'kx': kx, 'ky': ky, 't': t, 'dt': dt}


@Graph.node()
def time_dim():
    return hv.Dimension('time', label='time', unit='ms')


@Graph.node()
def frequency_dim():
    return hv.Dimension('frequency', label='G read', unit='mT/m', range=(-30, 30))


@Graph.node()
def phase_dim():
    return hv.Dimension('phase', label='G phase', unit='mT/m', range=(-30, 30))


@Graph.node()
def slice_dim():
    return hv.Dimension('slice', label='G slice', unit='mT/m', range=(-30, 30))


@Graph.node()
def RF_dim():
    return hv.Dimension('RF', label='RF', unit='μT', range=(-5, 25))


@Graph.node()
def signal_dim():
    return hv.Dimension('signal', label='signal', unit='a.u.', range=(-1, 1))


@Graph.node()
def ADC_dim():
    return hv.Dimension('ADC', label='ADC', unit='')


@Graph.node()
def frequency_objects(read_prephaser, readouts):
    objects = [read_prephaser, *flatten_dicts(readouts)]
    return [obj for obj in objects if obj]


@Graph.node()
def phase_objects(phasers, rephasers, blips):
    objects = [*flatten_dicts(phasers), *flatten_dicts(rephasers), *flatten_dicts(blips)]
    return [obj for obj in objects if obj]


@Graph.node()
def slice_objects(slice_select_inversion, inversion_spoiler, FatSat_spoiler, slice_select_excitation, slice_select_rephaser, slice_select_refocusing, spoiler):
    objects = [slice_select_inversion, inversion_spoiler, FatSat_spoiler, slice_select_excitation, slice_select_rephaser, *flatten_dicts(slice_select_refocusing), spoiler]
    return [obj for obj in objects if obj]


@Graph.node()
def RF_objects(RF_inversion, RF_FatSat, RF_excitation, RF_refocusing):
    objects = [RF_inversion, RF_FatSat, RF_excitation, *flatten_dicts(RF_refocusing)]
    return [obj for obj in objects if obj]


@Graph.node()
def signal_objects(signal_curves):
    return flatten_dicts(signal_curves)


@Graph.node()
def ADC_objects(sampling_windows):
    objects = flatten_dicts(sampling_windows)
    for obj in objects:
        obj.update({'c1': obj['time'][0], 'c2': -2, 'c3': obj['time'][-1], 'c4': 2})
    return objects


@Graph.node()
def TR_span(sequence_start, TR, time_dim, frequency_dim, phase_dim, slice_dim, RF_dim, signal_dim):
    TR_span = {}
    for board_dim in [frequency_dim, phase_dim, slice_dim, RF_dim, signal_dim]:
        TR_span[board_dim.name] = hv.VSpan(-20000, sequence_start, kdims=[time_dim, board_dim]).opts(color='gray', fill_alpha=.3)
        TR_span[board_dim.name] *= hv.VSpan(sequence_start + TR, 20000, kdims=[time_dim, board_dim]).opts(color='gray', fill_alpha=.3)
    return TR_span


@Graph.node()
def frequency_board(time_dim, frequency_dim, frequency_objects, TR_span, frequency_hover):
    vdims = [tip[0] for tip in frequency_hover.tooltips]
    specs = {'zero_line': hline(time_dim, frequency_dim),
                'net_gradient': hv.Area(sequence.accumulate_waveforms(frequency_objects, 'frequency'), time_dim, frequency_dim).opts(color=constants.BOARD_COLORS['frequency']),
                'waveforms': hv.Polygons(frequency_objects, kdims=[time_dim, frequency_dim], vdims=vdims).opts(tools=[frequency_hover], cmap=[constants.BOARD_COLORS['frequency']], hooks=[hideframe_hook, partial(bounds_hook, xbounds=(-19000, 19000))]),
                'TR_span': TR_span['frequency']}
    return specs


@Graph.node()
def phase_board(time_dim, phase_dim, phase_objects, TR_span, phase_hover):
    vdims = [tip[0] for tip in phase_hover.tooltips]
    specs = {'zero_lines': hline(time_dim, phase_dim),
                'net_gradient': hv.Area(sequence.accumulate_waveforms(phase_objects, 'phase'), time_dim, phase_dim).opts(color=constants.BOARD_COLORS['phase']),
                'waveforms': hv.Polygons(phase_objects, kdims=[time_dim, phase_dim], vdims=vdims).opts(tools=[phase_hover], cmap=[constants.BOARD_COLORS['phase']], hooks=[hideframe_hook, partial(bounds_hook, xbounds=(-19000, 19000))]),
                'TR_span': TR_span['phase']}
    return specs


@Graph.node()
def slice_board(time_dim, slice_dim, slice_objects, TR_span, slice_hover):
    vdims = [tip[0] for tip in slice_hover.tooltips]
    specs = {'zero_lines': hline(time_dim, slice_dim),
                'net_gradient': hv.Area(sequence.accumulate_waveforms(slice_objects, 'slice'), time_dim, slice_dim).opts(color=constants.BOARD_COLORS['slice']),
                'waveforms': hv.Polygons(slice_objects, kdims=[time_dim, slice_dim], vdims=vdims).opts(tools=[slice_hover], cmap=[constants.BOARD_COLORS['slice']], hooks=[hideframe_hook, partial(bounds_hook, xbounds=(-19000, 19000))]),
                'TR_span': TR_span['slice']}
    return specs


@Graph.node()
def RF_board(time_dim, RF_dim, RF_objects, TR_span, RF_hover):
    vdims = [tip[0] for tip in RF_hover.tooltips]
    specs = {'zero_lines': hline(time_dim, RF_dim),
                'net_RF': hv.Area(sequence.accumulate_waveforms(RF_objects, 'RF'), time_dim, RF_dim).opts(color=constants.BOARD_COLORS['RF']),
                'waveforms': hv.Polygons(RF_objects, kdims=[time_dim, RF_dim], vdims=vdims).opts(tools=[RF_hover], cmap=[constants.BOARD_COLORS['RF']], hooks=[hideframe_hook, partial(bounds_hook, xbounds=(-19000, 19000))]),
                'TR_span': TR_span['RF']}
    return specs


@Graph.node()
def signal_board(time_dim, signal_dim, signal_objects, ADC_objects, TR_span, signal_hover):
    vdims = [tip[0] for tip in signal_hover.tooltips]
    specs = {'zero_lines': hline(time_dim, signal_dim),
                'net_signal': hv.Area(sequence.accumulate_waveforms(signal_objects, 'signal'), time_dim, signal_dim).opts(color=constants.BOARD_COLORS['signal']),
                'waveforms': hv.Polygons(signal_objects, kdims=[time_dim, signal_dim], vdims='signal').opts(tools=[], cmap=[constants.BOARD_COLORS['signal']], hooks=[hideframe_hook, partial(bounds_hook, xbounds=(-19000, 19000))]),
                'sampling_windows': hv.Rectangles(ADC_objects, kdims=['c1', 'c2', 'c3', 'c4'], vdims=vdims).opts(tools=[signal_hover]),
                'TR_span': TR_span['signal']}
    return specs


@Graph.node()
def sequence_plot(frequency_board, phase_board, slice_board, RF_board, signal_board):
    boards = [frequency_board, phase_board, slice_board, RF_board, signal_board]
    board_plots = []
    for board in boards:
        last = board is boards[-1]
        board_plots.append(hv.Overlay(board.values()).opts(width=1700, height=180 if last else 120, border=0, xaxis='bottom' if last else None))
    return hv.Layout(list(board_plots)).cols(1).options(toolbar='below')


@Graph.node()
def kspace(kspace_type, show_processed_kspace, oversampled_recon_matrix, FOV, recon_matrix, full_k_matrix, zerofilled_kspace, kspace_exponent, gridded_kspace, k_grid_axes):
    operator = constants.OPERATORS[kspace_type]
    if show_processed_kspace:
        k_axes = []
        for dim in range(2):
            k_axes.append(recon.get_k_axis(oversampled_recon_matrix[dim], FOV[dim] / recon_matrix[dim]))
            # half-sample shift axis when odd number of zeroes:
            if (oversampled_recon_matrix[dim] - full_k_matrix[dim])%2:
                shift = recon_matrix[dim] / (2 * oversampled_recon_matrix[dim] * FOV[dim])
                k_axes[-1] -= shift
        ksp = xr.DataArray(
            operator(zerofilled_kspace**kspace_exponent), 
            dims=('ky', 'kx'),
            coords={'kx': k_axes[1], 'ky': k_axes[0]}
        )
    else:
        ksp = xr.DataArray(
            operator(gridded_kspace**kspace_exponent), 
            dims=('ky', 'kx'),
            coords={'kx': k_grid_axes[1], 'ky': k_grid_axes[0]}
        )
    ksp.kx.attrs['units'] = ksp.ky.attrs['units'] = '1/mm'
    lim = 1.12 * max(k_grid_axes[1])
    return hv.Image(ksp, vdims=['magnitude']).opts(xlim=(-lim,lim), ylim=(-lim,lim))


@Graph.node()
def FOV_box(show_FOV, is_radial, FOV, matrix, freq_dir, phase_dir, k_read_axis, k_phase_axis):
    if not show_FOV:
        return hv.Overlay([])
    rec_FOV_shape = hv.Box(0, 0, tuple(FOV[::-1])).opts(color='yellow')
    if is_radial:
        radial_FOV = FOV[freq_dir] * len(k_read_axis) / matrix[freq_dir]
        acq_FOV_shape = hv.Ellipse(0, 0, radial_FOV).opts(line_color='lightblue')
    else:
        acq_FOV = copy.deepcopy(FOV)
        acq_FOV[phase_dir] *= len(k_phase_axis) / matrix[phase_dir]
        acq_FOV_shape = hv.Box(0, 0, tuple(acq_FOV[::-1])).opts(color='lightblue')
    return acq_FOV_shape * rec_FOV_shape


@Graph.node()
def image(image_type, recon_matrix, FOV, image_array):
    operator = constants.OPERATORS[image_type]
    axes = [(np.arange(recon_matrix[dim]) - (recon_matrix[dim]-1)/2) / recon_matrix[dim] * FOV[dim] for dim in range(2)]
    img = xr.DataArray(
        operator(image_array), 
        dims=('y', 'x'),
        coords={'x': axes[1], 'y': axes[0][::-1]}
    )
    img.x.attrs['units'] = img.y.attrs['units'] = 'mm'
    return hv.Overlay([hv.Image(img, vdims=['magnitude'])])