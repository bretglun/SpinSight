import param
from spinsight.params import PARAMS


def init_param(par_name):
    return PARAMS[par_name].type(**PARAMS[par_name].param_kwargs)


class InputParams(param.Parameterized):

    # Settings
    object = init_param('object')
    field_strength = init_param('field_strength')
    parameter_style = init_param('parameter_style')

    min_voxel_size = init_param('min_voxel_size')
    noise_gain = init_param('noise_gain')

    # Sequence
    sequence_type = init_param('sequence_type')
    pixel_bandwidth = init_param('pixel_bandwidth')
    FOV_bandwidth = init_param('FOV_bandwidth')
    FW_shift = init_param('FW_shift')
    NSA = init_param('NSA')
    partial_Fourier = init_param('partial_Fourier')
    turbo_factor = init_param('turbo_factor')
    EPI_factor = init_param('EPI_factor')
    echo_order = init_param('echo_order')

    # Contrast
    FatSat = init_param('FatSat')
    TR = init_param('TR')
    TE = init_param('TE')
    TI = init_param('TI')
    FA = init_param('FA')
    
    # Geometry
    trajectory = init_param('trajectory')
    frequency_direction = init_param('frequency_direction')
    FOV_P = init_param('FOV_P')
    FOV_F = init_param('FOV_F')
    phase_oversampling = init_param('phase_oversampling')
    radial_oversampling = init_param('radial_oversampling')
    matrix_P = init_param('matrix_P')
    matrix_F = init_param('matrix_F')
    voxel_P = init_param('voxel_P')
    voxel_F = init_param('voxel_F')
    recon_matrix_P = init_param('recon_matrix_P')
    recon_matrix_F = init_param('recon_matrix_F')
    recon_voxel_P = init_param('recon_voxel_P')
    recon_voxel_F = init_param('recon_voxel_F')
    slice_thickness = init_param('slice_thickness')
    
    radial_FOV_oversampling = init_param('radial_FOV_oversampling')
    
    # MR image
    show_FOV = init_param('show_FOV')
    reference_tissue = init_param('reference_tissue')
    image_mode = init_param('image_mode')

    # k-space
    show_processed_kspace = init_param('show_processed_kspace')
    kspace_exponent = init_param('kspace_exponent')
    kspace_mode = init_param('kspace_mode')

    # Post-processing
    homodyne = init_param('homodyne')
    do_apodize = init_param('do_apodize')
    apodization_alpha = init_param('apodization_alpha')
    do_zerofill = init_param('do_zerofill')
    
    # Sequence plot
    shot = init_param('shot')
    signal_exponent = init_param('signal_exponent')