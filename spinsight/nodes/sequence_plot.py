from spinsight import sequence
from spinsight.constants import ACTION
from spinsight.styles import BOARD_COLORS, BOARD_PLOT_WIDTH, BOARD_PLOT_HEIGHT, BOARD_PLOT_HEIGHT_LAST, TIME_BOUNDS, G_READ_RANGE, G_PHASE_RANGE, G_SLICE_RANGE, RF_RANGE, SIGNAL_RANGE
from spinsight.DAG import Graph
import holoviews as hv
from functools import partial
import numpy as np


@Graph.node()
def sequence_plot(frequency_board, phase_board, slice_board, RF_board, signal_board):
    boards = [frequency_board, phase_board, slice_board, RF_board, signal_board]
    board_plots = []
    for board in boards:
        last = board is boards[-1]
        board_plots.append(hv.Overlay(board.values()).opts(width=BOARD_PLOT_WIDTH, height=BOARD_PLOT_HEIGHT if last else BOARD_PLOT_HEIGHT_LAST, border=0, xaxis='bottom' if last else None))
    return hv.Layout(list(board_plots)).cols(1).options(toolbar='below')


@Graph.node(action=ACTION.SEQPLOT)
def update_seqplot(controller, sequence_plot):
    controller.sequence_plot = sequence_plot


@Graph.node()
def frequency_board(time_dim, frequency_dim, frequency_objects, TR_span, frequency_hover):
    vdims = [tip[0] for tip in frequency_hover.tooltips]
    specs = {'zero_line': hline(time_dim, frequency_dim),
                'net_gradient': hv.Area(sequence.accumulate_waveforms(frequency_objects, 'frequency'), time_dim, frequency_dim).opts(color=BOARD_COLORS['frequency']),
                'waveforms': hv.Polygons(frequency_objects, kdims=[time_dim, frequency_dim], vdims=vdims).opts(tools=[frequency_hover], cmap=[BOARD_COLORS['frequency']], hooks=[hideframe_hook, partial(bounds_hook, xbounds=TIME_BOUNDS)]),
                'TR_span': TR_span['frequency']}
    return specs


@Graph.node()
def phase_board(time_dim, phase_dim, phase_objects, TR_span, phase_hover):
    vdims = [tip[0] for tip in phase_hover.tooltips]
    specs = {'zero_lines': hline(time_dim, phase_dim),
                'net_gradient': hv.Area(sequence.accumulate_waveforms(phase_objects, 'phase'), time_dim, phase_dim).opts(color=BOARD_COLORS['phase']),
                'waveforms': hv.Polygons(phase_objects, kdims=[time_dim, phase_dim], vdims=vdims).opts(tools=[phase_hover], cmap=[BOARD_COLORS['phase']], hooks=[hideframe_hook, partial(bounds_hook, xbounds=TIME_BOUNDS)]),
                'TR_span': TR_span['phase']}
    return specs


@Graph.node()
def slice_board(time_dim, slice_dim, slice_objects, TR_span, slice_hover):
    vdims = [tip[0] for tip in slice_hover.tooltips]
    specs = {'zero_lines': hline(time_dim, slice_dim),
                'net_gradient': hv.Area(sequence.accumulate_waveforms(slice_objects, 'slice'), time_dim, slice_dim).opts(color=BOARD_COLORS['slice']),
                'waveforms': hv.Polygons(slice_objects, kdims=[time_dim, slice_dim], vdims=vdims).opts(tools=[slice_hover], cmap=[BOARD_COLORS['slice']], hooks=[hideframe_hook, partial(bounds_hook, xbounds=TIME_BOUNDS)]),
                'TR_span': TR_span['slice']}
    return specs


@Graph.node()
def RF_board(time_dim, RF_dim, RF_objects, TR_span, RF_hover):
    vdims = [tip[0] for tip in RF_hover.tooltips]
    specs = {'zero_lines': hline(time_dim, RF_dim),
                'net_RF': hv.Area(sequence.accumulate_waveforms(RF_objects, 'RF'), time_dim, RF_dim).opts(color=BOARD_COLORS['RF']),
                'waveforms': hv.Polygons(RF_objects, kdims=[time_dim, RF_dim], vdims=vdims).opts(tools=[RF_hover], cmap=[BOARD_COLORS['RF']], hooks=[hideframe_hook, partial(bounds_hook, xbounds=TIME_BOUNDS)]),
                'TR_span': TR_span['RF']}
    return specs


@Graph.node()
def signal_board(time_dim, signal_dim, signal_objects, ADC_objects, TR_span, signal_hover):
    vdims = [tip[0] for tip in signal_hover.tooltips]
    specs = {'zero_lines': hline(time_dim, signal_dim, yticks=list(zip([0], ' '))),
                'net_signal': hv.Area(sequence.accumulate_waveforms(signal_objects, 'signal'), time_dim, signal_dim).opts(color=BOARD_COLORS['signal']),
                'waveforms': hv.Polygons(signal_objects, kdims=[time_dim, signal_dim], vdims='signal').opts(tools=[], cmap=[BOARD_COLORS['signal']], hooks=[hideframe_hook, partial(bounds_hook, xbounds=TIME_BOUNDS)]),
                'sampling_windows': hv.Rectangles(ADC_objects, kdims=['c1', 'c2', 'c3', 'c4'], vdims=vdims).opts(tools=[signal_hover]),
                'TR_span': TR_span['signal']}
    return specs


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


def hline(time_dim, amp_dim, yticks=None):
    return hv.HLine(0.0, kdims=[time_dim, amp_dim]).opts(tools=['xwheel_zoom', 'xpan', 'reset'], default_tools=[], active_tools=['xwheel_zoom', 'xpan'], yticks=yticks)


@Graph.node()
def TR_span(sequence_start, TR, time_dim, frequency_dim, phase_dim, slice_dim, RF_dim, signal_dim):
    TR_span = {}
    for board_dim in [frequency_dim, phase_dim, slice_dim, RF_dim, signal_dim]:
        TR_span[board_dim.name] = hv.VSpan(-20000, sequence_start, kdims=[time_dim, board_dim]).opts(color='gray', fill_alpha=.3)
        TR_span[board_dim.name] *= hv.VSpan(sequence_start + TR, 20000, kdims=[time_dim, board_dim]).opts(color='gray', fill_alpha=.3)
    return TR_span


@Graph.node()
def signal_curves(measured_kspace, shot, is_radial, turbo_factor, EPI_factor, pe_table, phase_dir, time_after_excitation, signal_exponent):
    signal_curves = []
    scale = 1 / np.max(np.abs(measured_kspace))
    spoke = shot if is_radial else 0
    for rf_echo in range(turbo_factor):
        signal_curves.append([])
        for gr_echo in range(EPI_factor):
            ky = pe_table[shot, rf_echo, gr_echo]
            waveform = np.real(np.take(measured_kspace[..., spoke], indices=ky, axis=phase_dir))
            t = np.take(time_after_excitation[..., spoke if spoke<time_after_excitation.shape[-1] else 0], indices=ky, axis=phase_dir)
            signal = sequence.get_signal(waveform, t, scale, signal_exponent)
            signal_curves[-1].append(signal)
    return signal_curves


def hideframe_hook(plot, elem):
    plot.handles['plot'].outline_line_color = None


def bounds_hook(plot, elem, xbounds=None):
    x_range = plot.handles['plot'].x_range
    if xbounds is not None:
        x_range.bounds = xbounds
    else:
        x_range.bounds = x_range.start, x_range.end 


@Graph.node()
def time_dim():
    return hv.Dimension('time', label='time', unit='ms')


@Graph.node()
def frequency_dim():
    return hv.Dimension('frequency', label='G read', unit='mT/m', range=G_READ_RANGE)


@Graph.node()
def phase_dim():
    return hv.Dimension('phase', label='G phase', unit='mT/m', range=G_PHASE_RANGE)


@Graph.node()
def slice_dim():
    return hv.Dimension('slice', label='G slice', unit='mT/m', range=G_SLICE_RANGE)


@Graph.node()
def RF_dim():
    return hv.Dimension('RF', label='RF', unit='μT', range=RF_RANGE)


@Graph.node()
def signal_dim():
    return hv.Dimension('signal', label='signal', unit='a.u.', range=SIGNAL_RANGE)


@Graph.node()
def ADC_dim():
    return hv.Dimension('ADC', label='ADC', unit='')


@Graph.node()
def frequency_hover(controller):
    return controller.get_hover_tool('frequency', ['name', 'center', 'duration', 'area'])


@Graph.node()
def phase_hover(controller):
    return controller.get_hover_tool('phase', ['name', 'center', 'duration', 'area'])


@Graph.node()
def slice_hover(controller):
    return controller.get_hover_tool('slice', ['name', 'center', 'duration', 'area'])


@Graph.node()
def RF_hover(controller):
    return controller.get_hover_tool('RF', ['name', 'center', 'duration', 'flip_angle'])


@Graph.node()
def signal_hover(controller):
    return controller.get_hover_tool('signal', ['name', 'center', 'duration'])


def flatten_dicts(list_of_dicts_and_lists):
    if list_of_dicts_and_lists is None:
        return []
    res = []
    for v in list_of_dicts_and_lists:
        res += flatten_dicts(v) if isinstance(v, list) else [v]
    return res