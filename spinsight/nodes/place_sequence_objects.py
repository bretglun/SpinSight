from spinsight import sequence
from spinsight.DAG import Graph
import copy


def place_waveform(waveform_floating, time):
    waveform = copy.deepcopy(waveform_floating)
    sequence.move_waveform(waveform, time)
    return waveform


@Graph.node()
def sequence_start(slice_select_inversion, RF_FatSat, slice_select_excitation):
    if slice_select_inversion is not None:
        return slice_select_inversion['time'][0]
    elif RF_FatSat is not None:
        return RF_FatSat['time'][0]
    else:
        return slice_select_excitation['time'][0]


@Graph.node()
def RF_inversion(RF_inversion_floating, TI):
    if RF_inversion_floating is None:
        return None
    return place_waveform(RF_inversion_floating, -TI)


@Graph.node()
def slice_select_inversion(slice_select_inversion_floating, TI):
    if slice_select_inversion_floating is None:
        return None
    return place_waveform(slice_select_inversion_floating, -TI)


@Graph.node()
def inversion_spoiler(inversion_spoiler_floating, RF_inversion):
    if inversion_spoiler_floating is None:
        return None
    time = RF_inversion['time'][-1] + inversion_spoiler_floating['dur_f']/2
    return place_waveform(inversion_spoiler_floating, time)


@Graph.node()
def RF_FatSat(RF_FatSat_floating, FatSat_spoiler_floating):
    if RF_FatSat_floating is None:
        return None
    time = FatSat_spoiler_floating['time'][0] - RF_FatSat_floating['dur_f']/2
    return place_waveform(RF_FatSat_floating, time)


@Graph.node()
def FatSat_spoiler(FatSat_spoiler_floating, slice_select_excitation):
    if FatSat_spoiler_floating is None:
        return None
    time = slice_select_excitation['time'][0] - FatSat_spoiler_floating['dur_f']/2
    return place_waveform(FatSat_spoiler_floating, time)


# RF_excitation and slice_select_excitation are created at time=0 and do not need to be placed


@Graph.node()
def slice_select_rephaser(slice_select_excitation, slice_select_rephaser_floating):
    time = (slice_select_excitation['dur_f'] + slice_select_rephaser_floating['dur_f']) / 2
    return place_waveform(slice_select_rephaser_floating, time)


@Graph.node()
def read_prephaser(read_prephaser_floating, is_gradient_echo, readouts, RF_excitation):
    if is_gradient_echo:
        first_readout = readouts[0][0]
        time = first_readout['center_f'] - (read_prephaser_floating['dur_f'] + first_readout['dur_f']) / 2
    else:
        time = (RF_excitation['dur_f'] + read_prephaser_floating['dur_f']) / 2
    return place_waveform(read_prephaser_floating, time)


@Graph.node()
def RF_refocusing(RF_refocusing_floating, refocusing_time):
    if RF_refocusing_floating is None:
        return None
    return [place_waveform(RF, time) for RF, time in zip(RF_refocusing_floating, refocusing_time)]


@Graph.node()
def slice_select_refocusing(slice_select_refocusing_floating, refocusing_time):
    if slice_select_refocusing_floating is None:
        return None
    return [place_waveform(grad, time) for grad, time in zip(slice_select_refocusing_floating, refocusing_time)]


@Graph.node()
def phasers(readtrain_center_time, phasers_floating, gre_echo_train_dur, readout_risetime):
    return [place_waveform(phaser, center - (gre_echo_train_dur + phaser['dur_f'])/2 + readout_risetime) for phaser, center in zip(phasers_floating, readtrain_center_time)]


@Graph.node()
def readouts(readouts_floating, readout_center_time):
    return [[place_waveform(readout, time) for readout, time in zip(readouts, times)] for readouts, times in zip(readouts_floating, readout_center_time)]


@Graph.node()
def sampling_windows(sampling_windows_floating, readout_center_time):
    return [[place_waveform(sampling, time) for sampling, time in zip(samplings, times)] for samplings, times in zip(sampling_windows_floating, readout_center_time)]


@Graph.node()
def blips(readtrain_center_time, EPI_factor, gr_echo_spacing, blips_floating):
    return [[place_waveform(blip, center + gr_echo_spacing * (gre - EPI_factor/2 + 1)) for gre, blip in enumerate(blips)] for center, blips in zip(readtrain_center_time, blips_floating)]


@Graph.node()
def rephasers(readtrain_center_time, gre_echo_train_dur, readout_risetime, rephasers_floating):
    return [place_waveform(rephaser, center + (gre_echo_train_dur + rephaser['dur_f'])/2 - readout_risetime) for rephaser, center in zip(rephasers_floating, readtrain_center_time)]


@Graph.node()
def spoiler(readouts, spoiler_floating):
    time = readouts[-1][-1]['center_f'] + (readouts[-1][-1]['flat_dur_f'] + spoiler_floating['dur_f']) / 2
    return place_waveform(spoiler_floating, time)