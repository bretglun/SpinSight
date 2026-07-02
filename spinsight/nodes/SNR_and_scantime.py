from spinsight.constants import ACTION
from spinsight.DAG import Graph
import numpy as np


@Graph.node()
def SNR(reference_signal, noise_std):
    return reference_signal / noise_std


@Graph.node()
def relative_SNR(reference_SNR, SNR):
    if not reference_SNR:
        return 1.0
    return SNR / reference_SNR


@Graph.node(action=ACTION.VALUE)
def set_reference_SNR(update_reference_SNR, controller, SNR):
    if update_reference_SNR:
        controller.reference_SNR = SNR
        controller.update_reference_SNR = False


@Graph.node()
def reference_signal(decayed_signal, PD_and_T1w, reference_tissue):
    return decayed_signal * np.abs(PD_and_T1w[reference_tissue])


@Graph.node()
def decayed_signal(signal_level, T2w, reference_tissue, k_read_axis, k_phase_axis, freq_dir):
    return signal_level * np.take(np.take(T2w[reference_tissue], np.argmin(np.abs(k_read_axis)), axis=freq_dir), np.argmin(np.abs(k_phase_axis)))


@Graph.node()
def signal_level(k_read_axis, num_sampled_phase_encodes, num_blades, slice_thickness, FOV, matrix):
    return np.sqrt(len(k_read_axis) * num_sampled_phase_encodes * num_blades) * slice_thickness * np.prod(FOV) / np.prod(matrix)


@Graph.node()
def scantime(num_shots, NSA, TR):
    return num_shots * NSA * TR