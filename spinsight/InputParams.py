import param
from spinsight.params import PARAMS


def init_param(func, par_name):
    return func(**PARAMS[par_name].param_kwargs)


class InputParams(param.Parameterized):

    # Settings
    object = init_param(param.ObjectSelector, 'object')
    field_strength = init_param(param.ObjectSelector, 'field_strength')
    parameter_style = init_param(param.ObjectSelector, 'parameter_style')

    min_voxel_size = init_param(param.Number, 'min_voxel_size')
    noise_gain = init_param(param.Number, 'noise_gain')

    # Sequence
    sequence_type = init_param(param.ObjectSelector, 'sequence_type')
    pixel_bandwidth_ui = init_param(param.Selector, 'pixel_bandwidth_ui')
    FOV_bandwidth = init_param(param.Selector, 'FOV_bandwidth')
    FW_shift = init_param(param.Selector, 'FW_shift')
    NSA = init_param(param.Integer, 'NSA')
    partial_Fourier = init_param(param.Number, 'partial_Fourier')
    turbo_factor = init_param(param.Integer, 'turbo_factor')
    EPI_factor = init_param(param.Selector, 'EPI_factor')

    # Contrast
    FatSat = init_param(param.Boolean, 'FatSat')
    TR_ui = init_param(param.Selector, 'TR_ui')
    TE_ui = init_param(param.Selector, 'TE_ui')
    TI = init_param(param.Selector, 'TI')
    FA = init_param(param.Selector, 'FA')
    
    # Geometry
    trajectory = init_param(param.ObjectSelector, 'trajectory')
    frequency_direction = init_param(param.ObjectSelector, 'frequency_direction')
    FOV_P = init_param(param.Selector, 'FOV_P')
    FOV_F = init_param(param.Selector, 'FOV_F')
    phase_oversampling = init_param(param.Selector, 'phase_oversampling')
    radial_oversampling = init_param(param.Selector, 'radial_oversampling')
    matrix_P_ui = init_param(param.Selector, 'matrix_P_ui')
    matrix_F_ui = init_param(param.Selector, 'matrix_F_ui')
    voxel_P = init_param(param.Selector, 'voxel_P')
    voxel_F = init_param(param.Selector, 'voxel_F')
    recon_matrix_P_ui = init_param(param.Selector, 'recon_matrix_P_ui')
    recon_matrix_F_ui = init_param(param.Selector, 'recon_matrix_F_ui')
    recon_voxel_P = init_param(param.Selector, 'recon_voxel_P')
    recon_voxel_F = init_param(param.Selector, 'recon_voxel_F')
    slice_thickness = init_param(param.Selector, 'slice_thickness')
    
    radial_FOV_oversampling = init_param(param.Number, 'radial_FOV_oversampling')
    
    # MR image
    show_FOV = init_param(param.Boolean, 'show_FOV')
    reference_tissue = init_param(param.ObjectSelector, 'reference_tissue')

    image_type = init_param(param.ObjectSelector, 'image_type')

    # k-space
    show_processed_kspace = init_param(param.Boolean, 'show_processed_kspace')
    kspace_exponent = init_param(param.Number, 'kspace_exponent')
    kspace_type = init_param(param.ObjectSelector, 'kspace_type')

    # Post-processing
    homodyne = init_param(param.Boolean, 'homodyne')
    do_apodize = init_param(param.Boolean, 'do_apodize')
    apodization_alpha = init_param(param.Number, 'apodization_alpha')
    do_zerofill = init_param(param.Boolean, 'do_zerofill')
    
    # Sequence plot
    shot_ui = init_param(param.Integer, 'shot_ui')
    signal_exponent = init_param(param.Number, 'signal_exponent')