from spinsight import sequence
from spinsight.DAG import Graph
from spinsight.constants import GYRO, MAX_AMP, MAX_SLEW, INVERSION_THK_FACTOR, SPOILER_AREA
import numpy as np


@Graph.node()
def RF_inversion_floating(sequence_type):
    if not sequence_type == 'Inversion Recovery':
        return None
    return sequence.get_RF(flip_angle=180., dur=3., shape='hamming_sinc',  name='inversion')


@Graph.node()
def slice_select_inversion_floating(sequence_type, RF_inversion_floating, slice_thickness):
    if sequence_type != 'Inversion Recovery':
        return None
    flat_dur = RF_inversion_floating['dur_f']
    amp = RF_inversion_floating['FWHM_f'] / (INVERSION_THK_FACTOR * slice_thickness * GYRO)
    return sequence.get_gradient('slice', max_amp=amp, flat_dur=flat_dur, name='slice select inversion', max_slew=MAX_SLEW)


@Graph.node()
def inversion_spoiler_floating(sequence_type):
    if sequence_type != 'Inversion Recovery':
        return None
    return sequence.get_gradient('slice', total_area=SPOILER_AREA, name='inversion spoiler', max_amp=MAX_AMP, max_slew=MAX_SLEW)


@Graph.node()
def RF_FatSat_floating(FatSat, field_strength):
    if not FatSat:
        return None
    return sequence.get_RF(flip_angle=90, time=0., dur=30./field_strength, shape='hamming_sinc',  name='FatSat')


@Graph.node()
def FatSat_spoiler_floating(FatSat):
    if not FatSat:
        return None
    return sequence.get_gradient('slice', total_area=SPOILER_AREA, name='FatSat spoiler', max_amp=MAX_AMP, max_slew=MAX_SLEW)


@Graph.node()
def RF_excitation(FA, is_gradient_echo):
    flip_angle = FA if is_gradient_echo else 90.
    return sequence.get_RF(flip_angle=flip_angle, time=0., dur=3., shape='hamming_sinc',  name='excitation')


@Graph.node()
def slice_select_excitation(RF_excitation, slice_thickness):
    flat_dur = RF_excitation['dur_f']
    amp = RF_excitation['FWHM_f'] / (slice_thickness * GYRO)
    time = 0.
    return sequence.get_gradient('slice', time, max_amp=amp, flat_dur=flat_dur, name='slice select excitation', max_slew=MAX_SLEW)


@Graph.node()
def slice_select_rephaser_floating(slice_select_excitation):
    slice_rephaser_area = -slice_select_excitation['area_f']/2
    return sequence.get_gradient('slice', total_area=slice_rephaser_area, name='slice select rephaser', max_amp=MAX_AMP, max_slew=MAX_SLEW)


@Graph.node()
def read_prephaser_floating(readouts_floating, is_gradient_echo):
    read_prephaser = sequence.get_gradient('frequency', total_area=readouts_floating[0][0]['area_f']/2, name='read prephaser', max_amp=MAX_AMP, max_slew=MAX_SLEW)
    if is_gradient_echo:
        sequence.rescale_gradient(read_prephaser, -1)
    return read_prephaser


@Graph.node()
def RF_refocusing_floating(is_gradient_echo, turbo_factor):
    if is_gradient_echo:
        return None
    RF_refocusing = []
    for rf_echo in range(turbo_factor):
        name = f'refocusing{" " + str(rf_echo + 1) if turbo_factor > 1 else ""}'
        RF_refocusing.append(sequence.get_RF(flip_angle=180., dur=3., shape='hamming_sinc',  name=name))
    return RF_refocusing


@Graph.node()
def slice_select_refocusing_floating(RF_refocusing_floating, slice_thickness, turbo_factor):
    if RF_refocusing_floating is None:
        return None
    flat_dur = RF_refocusing_floating[0]['dur_f']
    amp = RF_refocusing_floating[0]['FWHM_f'] / (slice_thickness * GYRO)
    slice_select_refocusing = []
    for rf_echo in range(turbo_factor):
        name = f'slice select refocusing{" " + str(rf_echo + 1) if turbo_factor > 1 else ""}'
        slice_select_refocusing.append(sequence.get_gradient('slice', max_amp=amp, flat_dur=flat_dur, name=name, max_slew=MAX_SLEW))
    return slice_select_refocusing


@Graph.node()
def largest_phaser_area(k_phase_axis):
    return np.min(k_phase_axis) * 1e3 / GYRO # uTs/m


@Graph.node()
def phase_step_area(k_phase_axis):
    if len(k_phase_axis)==1:
        return 0
    return np.mean(np.diff(k_phase_axis)) * 1e3 / GYRO # uTs/m


@Graph.node()
def phasers_floating(turbo_factor, largest_phaser_area, pe_table, phase_step_area, shot):
    phasers = []
    for rf_echo in range(turbo_factor):
        phaser_area = largest_phaser_area + pe_table[shot, rf_echo, 0] * phase_step_area
        suffix = f' {rf_echo + 1}' if turbo_factor > 1 else ''
        phaser = sequence.get_gradient('phase', total_area=largest_phaser_area, name='phase encode'+suffix, max_amp=MAX_AMP, max_slew=MAX_SLEW)
        if abs(largest_phaser_area) > 1e-5:
            sequence.rescale_gradient(phaser, phaser_area / largest_phaser_area)
        phasers.append(phaser)
    return phasers


@Graph.node()
def readouts_floating(k_read_axis, pixel_bandwidth, matrix_F, FOV_F, turbo_factor, EPI_factor):
    pixel_size = (len(k_read_axis)-1) / len(k_read_axis) / (max(k_read_axis)-min(k_read_axis))
    flat_area = 1e3 / pixel_size / GYRO # uTs/m
    amp = pixel_bandwidth * matrix_F / (FOV_F * GYRO) # mT/m
    readouts = []
    for rf_echo in range(turbo_factor):
        readouts.append([])           
        for gr_echo in range(EPI_factor):
            suffix = ((" " if (turbo_factor > 1 or EPI_factor > 1) else "")
                    + (str(rf_echo + 1) if turbo_factor > 1 else "")
                    + ("." if (turbo_factor > 1 and EPI_factor > 1) else "")
                    + (str(gr_echo + 1) if EPI_factor > 1 else ""))
            readout = sequence.get_gradient('frequency', max_amp=amp, flat_area=flat_area, name='readout'+suffix, max_slew=MAX_SLEW)
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
def blips_floating(turbo_factor, EPI_factor, phase_step_area, pe_table, shot):
    blips = []
    for rf_echo in range(turbo_factor):
        blips.append([])
        for gr_echo in range(1, EPI_factor):
            blip_area = phase_step_area * (pe_table[shot, rf_echo, gr_echo] - pe_table[shot, rf_echo, gr_echo-1])
            blip = sequence.get_gradient('phase', total_area=blip_area, name='blip', max_amp=MAX_AMP, max_slew=MAX_SLEW)
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
        rephaser = sequence.get_gradient('phase', total_area=largest_phaser_area, name='rephaser'+suffix, max_amp=MAX_AMP, max_slew=MAX_SLEW)
        if abs(largest_phaser_area) > 1e-5:
            sequence.rescale_gradient(rephaser, rephaser_area / largest_phaser_area)
        rephasers.append(rephaser)
    return rephasers


@Graph.node()
def spoiler_floating():
    return sequence.get_gradient('slice', total_area=SPOILER_AREA, name='spoiler', max_amp=MAX_AMP, max_slew=MAX_SLEW)