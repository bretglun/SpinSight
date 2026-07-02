from spinsight import convert
from spinsight.DAG import Graph
from spinsight.params import PARAMS
from spinsight.constants import ACTION


@Graph.node(action=ACTION.BOUNDS)
def set_TI_bounds(controller, sequence_type, max_TI):
    if sequence_type == 'Inversion Recovery':
        controller.set_param_bounds('TI', maxval=max_TI)


@Graph.node(action=ACTION.BOUNDS)
def set_slice_thickness_bounds(controller, min_slice_thickness):
    controller.set_param_bounds('slice_thickness', minval=min_slice_thickness)


@Graph.node(action=ACTION.BOUNDS)
def set_pixel_bandwidth_bounds(controller, pixel_BW_is_input, pixel_bandwidth_bounds):
    if pixel_BW_is_input:
        controller.set_param_bounds('pixel_bandwidth_ui', minval=pixel_bandwidth_bounds.min, maxval=pixel_bandwidth_bounds.max)


@Graph.node(action=ACTION.BOUNDS)
def set_FOV_bandwidth_bounds(controller, FOV_BW_is_input, pixel_bandwidth_bounds, matrix_F):
    if FOV_BW_is_input:
        controller.set_param_bounds('FOV_bandwidth', minval=convert.pixel_BW_to_FOV_BW(pixel_bandwidth_bounds.min, matrix_F), maxval=convert.pixel_BW_to_FOV_BW(pixel_bandwidth_bounds.max, matrix_F))


@Graph.node(action=ACTION.BOUNDS)
def set_FW_shift_bounds(controller, FW_shift_is_input, pixel_bandwidth_bounds, field_strength):
    if FW_shift_is_input:
        controller.set_param_bounds('FW_shift', minval=convert.pixel_BW_to_shift(pixel_bandwidth_bounds.max, field_strength), maxval=convert.pixel_BW_to_shift(pixel_bandwidth_bounds.min, field_strength))


@Graph.node(action=ACTION.BOUNDS)
def set_matrix_F_bounds(controller, matrix_is_input, matrix_F_bounds):
    if matrix_is_input:
        controller.set_param_bounds('matrix_F_ui', minval=matrix_F_bounds.min, maxval=matrix_F_bounds.max)


@Graph.node(action=ACTION.BOUNDS)
def set_matrix_P_bounds(controller, matrix_is_input, matrix_P_bounds):
    if matrix_is_input:
        controller.set_param_bounds('matrix_P_ui', minval=matrix_P_bounds.min, maxval=matrix_P_bounds.max)


@Graph.node(action=ACTION.BOUNDS)
def set_recon_matrix_F_bounds(controller, matrix_is_input, recon_matrix_F_bounds):
    if matrix_is_input:
        controller.set_param_bounds('recon_matrix_F_ui', minval=recon_matrix_F_bounds.min, maxval=recon_matrix_F_bounds.max)


@Graph.node(action=ACTION.BOUNDS)
def set_recon_matrix_P_bounds(controller, matrix_is_input, recon_matrix_P_bounds):
    if matrix_is_input:
        controller.set_param_bounds('recon_matrix_P_ui', minval=recon_matrix_P_bounds.min, maxval=recon_matrix_P_bounds.max)


@Graph.node(action=ACTION.BOUNDS)
def set_FOV_F_bounds(controller, FOV_F_bounds):
    controller.set_param_bounds('FOV_F', minval=FOV_F_bounds.min, maxval=FOV_F_bounds.max)


@Graph.node(action=ACTION.BOUNDS)
def set_FOV_P_bounds(controller, FOV_P_bounds):
    controller.set_param_bounds('FOV_P', minval=FOV_P_bounds.min, maxval=FOV_P_bounds.max)


@Graph.node(action=ACTION.BOUNDS)
def set_voxel_F_bounds(controller, voxel_size_is_input, FOV_F, matrix_F_bounds):
    if voxel_size_is_input:
        controller.set_param_bounds('voxel_F', minval=FOV_F/matrix_F_bounds.max, maxval=FOV_F/matrix_F_bounds.min)


@Graph.node(action=ACTION.BOUNDS)
def set_voxel_P_bounds(controller, voxel_size_is_input, FOV_P, matrix_P_bounds):
    if voxel_size_is_input:
        controller.set_param_bounds('voxel_P', minval=FOV_P/matrix_P_bounds.max, maxval=FOV_P/matrix_P_bounds.min)


@Graph.node(action=ACTION.BOUNDS)
def set_recon_voxel_F_bounds(controller, voxel_size_is_input, FOV_F, recon_matrix_F_bounds):
    if voxel_size_is_input:
        controller.set_param_bounds('recon_voxel_F', minval=FOV_F/recon_matrix_F_bounds.max, maxval=FOV_F/recon_matrix_F_bounds.min)


@Graph.node(action=ACTION.BOUNDS)
def set_recon_voxel_P_bounds(controller, voxel_size_is_input, FOV_P, recon_matrix_P_bounds):
    if voxel_size_is_input:
        controller.set_param_bounds('recon_voxel_P', minval=FOV_P/recon_matrix_P_bounds.max, maxval=FOV_P/recon_matrix_P_bounds.min)


@Graph.node(action=ACTION.BOUNDS)
def set_turbo_factor_bounds(controller, max_turbo_factor):
    # turbo_factor must equal 1 when the EPI_factor is even
    if not controller.input.EPI_factor%2:
        controller.input.param.turbo_factor.bounds = (1, 1)
        controller.input.param.turbo_factor.constant = True
        return
    controller.input.param.turbo_factor.bounds = (1, min(max_turbo_factor, PARAMS['turbo_factor'].bounds[-1]))
    controller.input.param.turbo_factor.constant = False


@Graph.node(action=ACTION.BOUNDS)
def set_EPI_factor_objects(controller, max_EPI_factor):
    controller.set_param_bounds('EPI_factor', maxval=max_EPI_factor)
    # EPI_factor must be odd for turbo spin echo (GRASE)
    if controller.input.turbo_factor > 1:
        controller.input.param.EPI_factor.objects = [v for v in controller.input.param.EPI_factor.objects if v%2]


@Graph.node(action=ACTION.BOUNDS)
def set_reference_tissue_objects(controller, tissues):
    controller.input.param.reference_tissue.objects = tissues
    controller.input.reference_tissue = tissues[0]


@Graph.node(action=ACTION.BOUNDS)
def set_x_y_labels(controller, frequency_direction):
    for p in ['FOV_F', 'FOV_P', 'matrix_F_ui', 'matrix_P_ui', 'recon_matrix_F_ui', 'recon_matrix_P_ui']:
        par = controller.input.param[p]
        if (' y' in par.label) and (('_F' in par.name and frequency_direction=='left-right') or
                                    ('_P' in par.name and frequency_direction=='anterior-posterior')):
            par.label = par.label.replace(' y', ' x')
        elif (' x' in par.label) and (('_P' in par.name and frequency_direction=='left-right') or
                                    ('_F' in par.name and frequency_direction=='anterior-posterior')):
            par.label = par.label.replace(' x', ' y')


@Graph.node()
def shot_label(is_radial, EPI_factor, turbo_factor):
    return 'shot' if not is_radial else 'spoke' if (EPI_factor * turbo_factor == 1) else 'blade'


@Graph.node(action=ACTION.BOUNDS)
def set_shot_label(controller, shot_label):
    controller.shot_label = shot_label
    controller.input.param.shot_ui.label = f'Displayed {shot_label}'