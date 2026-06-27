from spinsight.DAG import Graph
from spinsight.constants import ACTION, OPERATORS
import holoviews as hv
import numpy as np
import xarray as xr
import copy


@Graph.node()
def annotated_image(image, FOV_box):
    return image * FOV_box


@Graph.node(action=ACTION.IMAGE)
def update_image(simulator, annotated_image):
    simulator.image_update += 1


@Graph.node()
def image(image_type, recon_matrix, FOV, image_array):
    operator = OPERATORS[image_type]
    axes = [(np.arange(recon_matrix[dim]) - (recon_matrix[dim]-1)/2) / recon_matrix[dim] * FOV[dim] for dim in range(2)]
    img = xr.DataArray(
        operator(image_array), 
        dims=('y', 'x'),
        coords={'x': axes[1], 'y': axes[0][::-1]}
    )
    img.x.attrs['units'] = img.y.attrs['units'] = 'mm'
    return hv.Overlay([hv.Image(img, vdims=['magnitude'])])


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