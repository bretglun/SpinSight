from spinsight import sequence
from spinsight.DAG import Graph
from spinsight.constants import MAX_AMP, MAX_SLEW


def get_readtrain_spacing(TE, num_gr_echoes, gr_echo_spacing, k0_gr_echo_index, k0_rf_echo_index):
    readtrain_center = TE - readtrain_shift(gr_echo_spacing, k0_gr_echo_index, num_gr_echoes)
    return readtrain_center / (1 + k0_rf_echo_index)


def readtrain_shift(gr_echo_spacing, k0_gr_echo_index, num_gr_echoes):
    return gr_echo_spacing * (k0_gr_echo_index - (num_gr_echoes - 1) / 2)


def min_readtrain_spacing(gr_echo_spacing, k0_gr_echo_index, num_gr_echoes, is_gradient_echo, min_RF_to_readtrain_center, num_rf_echoes, min_refocusing_time):
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


@Graph.node()
def readtrain_spacing(EPI_factor, gr_echo_spacing, TE, k0_gr_echo_index, k0_rf_echo_index):
    # Equals center position of gradient echo (train) for gradient echo sequences
    # Equals rf echo spacing for spin echo sequences
    return get_readtrain_spacing(TE, EPI_factor, gr_echo_spacing, k0_gr_echo_index, k0_rf_echo_index)


@Graph.node()
def readtrain_center_time(readtrain_spacing, turbo_factor):
    # center position of gradient echo readout (train)(s)
    return [readtrain_spacing * (rf_echo + 1) for rf_echo in range(turbo_factor)]


@Graph.node()
def refocusing_time(TE, readtrain_center_time, readtrain_spacing):
    if len(readtrain_center_time) == 1:
        return [TE / 2]
    return [t - readtrain_spacing / 2 for t in readtrain_center_time]


@Graph.node()
def readout_center_time(EPI_factor, gr_echo_spacing, readtrain_center_time):
    return [[center_time + (gre - (EPI_factor-1) / 2) * gr_echo_spacing for gre in range(EPI_factor)] for center_time in readtrain_center_time]


@Graph.node()
def gr_echo_spacing(readouts_floating, readout_gap):
    return readouts_floating[0][0]['dur_f'] + readout_gap


@Graph.node()
def gre_echo_train_dur(EPI_factor, gr_echo_spacing, readout_gap):
    return EPI_factor * gr_echo_spacing - readout_gap


@Graph.node()
def readout_gap(max_blip_dur, readouts_floating):
    # gap between readout gradients
    return max(max_blip_dur - 2 * readouts_floating[0][0]['risetime_f'], 0)


@Graph.node()
def readout_risetime(readouts_floating):
    return readouts_floating[0][0]['risetime_f']


@Graph.node()
def phaser_duration(largest_phaser_area):
    largest_phaser = sequence.get_gradient('phase', total_area=largest_phaser_area, max_amp=MAX_AMP, max_slew=MAX_SLEW)
    return largest_phaser['dur_f']


@Graph.node()
def max_blip_dur(EPI_factor, phase_step_area, num_shots, turbo_factor):
    if (EPI_factor <= 1):
        return 0
    max_blip_area = phase_step_area * num_shots * turbo_factor
    max_blip = sequence.get_gradient('phase', total_area=max_blip_area, max_amp=MAX_AMP, max_slew=MAX_SLEW)
    return max_blip['dur_f']