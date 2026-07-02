from spinsight.constants import GYRO, MAX_AMP, MAX_SLEW, INVERSION_THK_FACTOR
from spinsight.DAG import Graph
from spinsight.params import PARAMS
from spinsight.nodes.sequence_timing import readtrain_shift, min_readtrain_spacing
import numpy as np
from collections import namedtuple


MAX_PHASE_OVERSAMPLING_FACTOR = list(PARAMS['phase_oversampling'].objects.values())[-1]

MinMax = namedtuple('MinMax', ['min', 'max'])


@Graph.node()
def max_turbo_factor(matrix, phase_dir, partial_Fourier, EPI_factor, is_radial):
    undersampling = partial_Fourier if not is_radial else 1
    return int(np.floor(matrix[phase_dir] * undersampling / EPI_factor * MAX_PHASE_OVERSAMPLING_FACTOR))


@Graph.node()
def max_EPI_factor(matrix, phase_dir, partial_Fourier, turbo_factor, is_radial):
    undersampling = partial_Fourier if not is_radial else 1
    return int(np.floor(matrix[phase_dir] * undersampling / turbo_factor * MAX_PHASE_OVERSAMPLING_FACTOR))


@Graph.node()
def min_voxel_F(max_readout_area):
    return 1e3 / (max_readout_area * GYRO)


@Graph.node()
def min_voxel_P(max_phaser_area):
    return 1e3 / (2 * max_phaser_area * GYRO)


@Graph.node()
def matrix_F_bounds(voxel_size_is_input, min_voxel_F, FOV_F, FOV_BW_is_input, FOV_bandwidth, pixel_bandwidth_bounds):
    min_matrix_F = [PARAMS['matrix_F_ui'].objects[0]]
    max_matrix_F = [PARAMS['matrix_F_ui'].objects[-1]]
    if not voxel_size_is_input:
        max_matrix_F.append(FOV_F / min_voxel_F)
    if FOV_BW_is_input: # constant FOV BW puts contraints on matrix_F
        min_matrix_F.append(FOV_bandwidth * 2e3 / pixel_bandwidth_bounds.max)
        max_matrix_F.append(FOV_bandwidth * 2e3 / pixel_bandwidth_bounds.min)
    return MinMax(max(min_matrix_F), min(max_matrix_F))


@Graph.node()
def matrix_P_bounds(voxel_size_is_input, min_voxel_P, FOV_P):
    min_matrix_P = [PARAMS['matrix_P_ui'].objects[0]]
    max_matrix_P = [PARAMS['matrix_P_ui'].objects[-1]]
    if not voxel_size_is_input:
        max_matrix_P.append(FOV_P / min_voxel_P)
    return MinMax(max(min_matrix_P), min(max_matrix_P))


@Graph.node()
def recon_matrix_F_bounds(matrix_F):
    min_recon_matrix_F = [PARAMS['recon_matrix_F_ui'].objects[0]]
    max_recon_matrix_F = [PARAMS['recon_matrix_F_ui'].objects[-1]]
    min_recon_matrix_F.append(matrix_F)
    return MinMax(max(min_recon_matrix_F),min(max_recon_matrix_F))


@Graph.node()
def recon_matrix_P_bounds(matrix_P):
    min_recon_matrix_P = [PARAMS['recon_matrix_P_ui'].objects[0]]
    max_recon_matrix_P = [PARAMS['recon_matrix_P_ui'].objects[-1]]
    min_recon_matrix_P.append(matrix_P)
    return MinMax(max(min_recon_matrix_P),min(max_recon_matrix_P))


@Graph.node()
def FOV_F_bounds(matrix_F, min_voxel_F, voxel_size_is_input, voxel_F, matrix_F_bounds, recon_voxel_F, recon_matrix_F_bounds):
    min_FOV_F = [list(PARAMS['FOV_F'].objects.values())[0]]
    max_FOV_F = [list(PARAMS['FOV_F'].objects.values())[-1]]
    if voxel_size_is_input: # constant voxel size puts constraints on FOV
        min_FOV_F.append(voxel_F * matrix_F_bounds.min)
        max_FOV_F.append(voxel_F * matrix_F_bounds.max)
    else:
        min_FOV_F.append(min_voxel_F * matrix_F)
    return MinMax(max(min_FOV_F), min(max_FOV_F))


@Graph.node()
def FOV_P_bounds(matrix_P, min_voxel_P, voxel_size_is_input, matrix_P_bounds, voxel_P, recon_voxel_P, recon_matrix_P_bounds):
    min_FOV_P = [list(PARAMS['FOV_P'].objects.values())[0]]
    max_FOV_P = [list(PARAMS['FOV_P'].objects.values())[-1]]
    if voxel_size_is_input: # constant voxel size puts constraints on FOV
        min_FOV_P.append(voxel_P * matrix_P_bounds.min)
        max_FOV_P.append(voxel_P * matrix_P_bounds.max)
    else:
        min_FOV_P.append(min_voxel_P * matrix_P)
    return MinMax(max(min_FOV_P), min(max_FOV_P))


@Graph.node()
def min_TR(spoiler, sequence_start):
    return spoiler['time'][-1] - sequence_start


@Graph.node()
def max_TI(TR, spoiler, slice_select_inversion_floating):
    if slice_select_inversion_floating is None:
        return None
    return TR - spoiler['time'][-1] - slice_select_inversion_floating['dur_f'] / 2


@Graph.node()
def min_TE(k0_echo_indices_linear_order, k0_echo_indices_reverse_linear_order, min_refocusing_time, min_RF_to_readtrain_center, gr_echo_spacing, EPI_factor, is_gradient_echo, turbo_factor):
    if EPI_factor == 1:
        gr_index, rf_index = 0, 0
        return min_TE_from_k0_echo_indices(gr_echo_spacing, gr_index, EPI_factor, is_gradient_echo, min_RF_to_readtrain_center, turbo_factor, min_refocusing_time, rf_index)
    
    min_TE_cands = []
    for indices in [k0_echo_indices_linear_order, k0_echo_indices_reverse_linear_order]:
        for (gr_index, rf_index) in indices:
            min_TE_cands.append(min_TE_from_k0_echo_indices(gr_echo_spacing, gr_index, EPI_factor, is_gradient_echo, min_RF_to_readtrain_center, turbo_factor, min_refocusing_time, rf_index))
    return min(min_TE_cands)


def min_TE_from_k0_echo_indices(gr_echo_spacing, k0_gr_echo_index, num_gr_echoes, is_gradient_echo, min_RF_to_readtrain_center, num_rf_echoes, min_refocusing_time, k0_rf_echo_index):
    readtrain_spacing = min_readtrain_spacing(gr_echo_spacing, k0_gr_echo_index, num_gr_echoes, is_gradient_echo, min_RF_to_readtrain_center, num_rf_echoes, min_refocusing_time)
    min_spin_echo_time = readtrain_spacing * (1 + k0_rf_echo_index)
    return min_spin_echo_time + readtrain_shift(gr_echo_spacing, k0_gr_echo_index, num_gr_echoes)


@Graph.node()
def min_refocusing_time(is_gradient_echo, RF_excitation, RF_refocusing_floating, read_prephaser_floating, slice_select_excitation, slice_select_rephaser_floating, slice_select_refocusing_floating):
    # Get earliest position of refocusing pulse
    if is_gradient_echo:
        return 0
    return RF_excitation['dur_f'] / 2 + RF_refocusing_floating[0]['dur_f'] / 2 + max(
        read_prephaser_floating['dur_f'], 
        slice_select_excitation['risetime_f'] + slice_select_rephaser_floating['dur_f'] + (slice_select_refocusing_floating[0]['risetime_f'])
        )

@Graph.node()
def max_TE(TR, sequence_start, spoiler_floating, readout_risetime, gre_echo_train_dur, EPI_factor, turbo_factor, k0_echo_indices_linear_order, k0_echo_indices_reverse_linear_order, gr_echo_spacing):
    last_readtrain_center = TR + sequence_start - spoiler_floating['dur_f'] + readout_risetime - gre_echo_train_dur / 2
    if EPI_factor == 1:
        return last_readtrain_center
    
    readtrain_spacing = last_readtrain_center / turbo_factor
    max_TE_cands = []
    for indices in [k0_echo_indices_linear_order, k0_echo_indices_reverse_linear_order]:
        for (gr_index, rf_index) in indices:
            readtrain_center = readtrain_spacing * (rf_index + 1)
            TE = readtrain_center + readtrain_shift(gr_echo_spacing, gr_index, EPI_factor)
            max_TE_cands.append(TE)
    return max(max_TE_cands)


@Graph.node()
def min_RF_to_readtrain_center(is_gradient_echo, RF_excitation, read_prephaser_floating, slice_select_excitation, slice_select_rephaser_floating, RF_refocusing_floating, slice_select_refocusing_floating, gre_echo_train_dur, readout_risetime, phaser_duration):
    # Get shortest spacing bewteen RF (excitation or refocusing for gradient / spin echo) and center of readout (train)
    if is_gradient_echo:
        RF_dur = RF_excitation['dur_f']
        read_prephaser_dur = read_prephaser_floating['dur_f']
        slice_rewind_dur = slice_select_excitation['risetime_f'] + slice_select_rephaser_floating['dur_f']
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
def pixel_bandwidth_bounds(matrix_F, FOV_F, is_gradient_echo, RF_refocusing, turbo_factor, EPI_factor, TE, RF_excitation, refocusing_time, readtrain_spacing, TR, sequence_start, spoiler, k0_echo_indices_linear_order, k0_echo_indices_reverse_linear_order, reverse_linear_order, read_prephaser, phaser_duration, slice_select_excitation, slice_select_rephaser_floating, max_blip_dur, slice_select_refocusing):
    readout_area = 1e3 * matrix_F / (FOV_F * GYRO)
    # min limit imposed by maximum gradient amplitude:
    min_read_duration = readout_area / MAX_AMP

    first_readtrain_ref = TE if (turbo_factor == 1) else readtrain_spacing
    last_readtrain_ref = TE if (turbo_factor == 1) else readtrain_spacing * turbo_factor
    last_read_end = TR + sequence_start - spoiler['dur_f']
    max_dur_ref_to_read_end = last_read_end - last_readtrain_ref
    
    slice_select_rewind_dur = slice_select_excitation['risetime_f'] + slice_select_rephaser_floating['dur_f'] if is_gradient_echo else slice_select_refocusing[0]['risetime_f']

    k0_echo_indices = k0_echo_indices_linear_order if reverse_linear_order else k0_echo_indices_reverse_linear_order
    
    pixel_bandwidth_values = []
    for pixel_bandwidth in PARAMS['pixel_bandwidth_ui'].objects.values():
        
        read_duration = 1e3 / pixel_bandwidth
        
        if read_duration < min_read_duration:
            continue
        
        readout_risetime = readout_area / read_duration / MAX_SLEW
        read_gap = max(max_blip_dur, 2 * readout_risetime)
        
        for (gr_echo, _) in k0_echo_indices:
            num_blips_before_ref = gr_echo if (turbo_factor == 1) else (EPI_factor-1) / 2
            num_blips_after_ref = EPI_factor - 1 - num_blips_before_ref
            
            dur_ref_to_read_end = read_duration * (num_blips_after_ref + 1/2) + read_gap * num_blips_after_ref
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
            dur_read_start_to_ref = read_duration * (num_blips_before_ref + 1/2) + read_gap * num_blips_before_ref
            
            if dur_read_start_to_ref > max_dur_read_start_to_ref:
                continue

            pixel_bandwidth_values.append(pixel_bandwidth)

    return MinMax(min(pixel_bandwidth_values, default=np.inf), max(pixel_bandwidth_values, default=-np.inf))


@Graph.node()
def min_slice_thickness(RF_excitation, is_gradient_echo, RF_refocusing, sequence_type, RF_inversion, TR, spoiler, sampling_windows):
    min_thks = [RF_excitation['FWHM_f'] / (MAX_AMP * GYRO)]
    if not is_gradient_echo:
        min_thks.append(RF_refocusing[0]['FWHM_f'] / (MAX_AMP * GYRO))
    if sequence_type == 'Inversion Recovery':
        min_thks.append(RF_inversion['FWHM_f'] / (MAX_AMP * GYRO) * INVERSION_THK_FACTOR)
    
    # Constraint due to TR: 
    if sequence_type == 'Inversion Recovery':
        max_risetime = TR - (spoiler['time'][-1] - RF_inversion['time'][0])
        max_amp = MAX_SLEW * max_risetime
        min_thks.append(RF_inversion['FWHM_f'] / (max_amp * GYRO))
    else:
        max_risetime = TR - (spoiler['time'][-1] - RF_excitation['time'][0])
        max_amp = MAX_SLEW * max_risetime
        min_thks.append(RF_excitation['FWHM_f'] / (max_amp * GYRO))
    
    # See paramBounds.tex for formulae
    s = MAX_SLEW
    d = RF_excitation['dur_f']
    if is_gradient_echo: # Constraint due to slice rephaser
        t = sampling_windows[0][0]['time'][0]
        h = s * (t - np.sqrt(t**2/2 + d**2/8))
        h = min(h, MAX_AMP)
        A = d * (np.sqrt((d*s+2*h)**2 - 8*h*(h-s*(t-d/2))) - d*s - 2*h) / 2
    else: # Spin echo: Constraint due to slice rephaser and refocusing slice select rampup
        t = RF_refocusing[0]['time'][0]
        h = s * (np.sqrt(2*(d + 2*t)**2 - 4*d**2) - d - 2*t) / 4
        h = min(h, MAX_AMP)
        A = (np.sqrt((d*(d*s + 4*h))**2 - 4*d**2*h*(d*s + 2*h - 2*s*t)) - d*(d*s + 4*h)) / 2
    Be = RF_excitation['FWHM_f']
    min_thks.append(Be * d / (GYRO * A)) # mm
    return max(min_thks)


@Graph.node()
def max_phaser_area(is_gradient_echo, readtrain_spacing, RF_excitation, gre_echo_train_dur, readout_risetime, refocusing_time, RF_refocusing):
    if is_gradient_echo:
        max_phaser_duration = readtrain_spacing - RF_excitation['dur_f']/2 - gre_echo_train_dur/2 + readout_risetime
    else:
        max_phaser_duration = readtrain_spacing - refocusing_time[0] - RF_refocusing[0]['dur_f'] / 2 - gre_echo_train_dur / 2 + readout_risetime
    max_risetime = MAX_AMP / MAX_SLEW
    if max_phaser_duration > 2 * max_risetime: # trapezoid maxPhaser
        max_phaserarea = (max_phaser_duration - max_risetime) * MAX_AMP
    else: # triangular maxPhaser
        max_phaserarea = (max_phaser_duration/2)**2 * MAX_SLEW
    return max_phaserarea


@Graph.node()
def max_readout_area(pixel_bandwidth, is_gradient_echo, k0_gr_echo_index, TE, RF_excitation, phaser_duration, slice_select_excitation, slice_select_rephaser_floating, max_blip_dur, readtrain_spacing, refocusing_time, RF_refocusing_floating, EPI_factor):
    max_readout_areas = []
    # See paramBounds.tex for formulae
    d = 1e3 / pixel_bandwidth # readout duration
    s = MAX_SLEW
    if is_gradient_echo:
        N = k0_gr_echo_index + 1/2
        M = k0_gr_echo_index * 2 + 1
        t = TE - RF_excitation['dur_f']/2
        v = 0 # gap between readouts
        for _ in range(2): # update readout gap after first pass
            if (M > 1):
                # max wrt G slice or G phase:
                q = t - max(phaser_duration,
                            slice_select_excitation['risetime_f'] + slice_select_rephaser_floating['dur_f'])
                A = d*s*(q - N*(d+v) + v/2) / (M-1) # eq. 15
                max_readout_areas.append(A)
            # max wrt G read:
            h_roots = np.roots([8*(3-2*M), 4*s*(t*(2*M-6)+d*(2*N-M)+v*(2*N-1)), s**2*(4*t**2-d**2)]) # eq. 12
            h = min([h for h in h_roots if h>0] + [MAX_AMP]) # truncate prephaser amp to max amp
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
        h = min(h, MAX_AMP)
        Ap = d * (np.sqrt((d*s)**2 - 8*h*(h-s*tp)) - d*s) / 2
        max_readout_areas.append(Ap)
    max_readout_areas.append(MAX_AMP * 1e3 / pixel_bandwidth) # max wrt max_amp
    return min(max_readout_areas)