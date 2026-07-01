from spinsight import recon
from spinsight.constants import ACTION, OPERATORS, GYRO
from spinsight.DAG import Graph
import holoviews as hv
import xarray as xr
import numpy as np


@Graph.node()
def kspace(kspace_type, show_processed_kspace, oversampled_recon_matrix, FOV, recon_matrix, full_k_matrix, zerofilled_kspace, kspace_exponent, gridded_kspace, k_grid_axes):
    operator = OPERATORS[kspace_type]
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


@Graph.node(action=ACTION.KSPACE)
def update_kspace(dashboard, kspace):
    dashboard.kspace = kspace


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


def get_k_coords(t, gp, tp, refocus_intervals):
    g = np.interp(t, tp, gp)
    dk = np.diff(t) * (g[:-1] + np.diff(g)/2) * GYRO * 1e-3
    k = np.insert(np.cumsum(dk), 0, 0.) # start at k=0
    for (ref_start, ref_stop) in refocus_intervals:
        # k inversion of refocusing pulse corresponds to negative shift of 2k:
        k_before = k[t<=ref_start][-1]
        refocus_times = t[(t>ref_start) & (t<ref_stop)]
        k[(t>ref_start) & (t<ref_stop)] -= 2 * k_before * (refocus_times - ref_start) / (ref_stop - ref_start)
        k[t>=ref_stop] -= 2 * k_before
    return k


@Graph.node(action=ACTION.KSPACE)
def update_k_trajectory(dashboard, k_trajectory):
    dashboard.hover.k_trajectory = k_trajectory


@Graph.node()
def spoke_angle(k_angles, shot):
    return np.degrees(k_angles[min(shot, len(k_angles)-1)])