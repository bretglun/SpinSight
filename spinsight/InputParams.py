import param
from spinsight.params import PARAMS


class InputParams(param.Parameterized):

    # Settings
    object = param.ObjectSelector(**PARAMS['object'].param_kwargs)
    field_strength = param.ObjectSelector(**PARAMS['field_strength'].param_kwargs)
    parameter_style = param.ObjectSelector(**PARAMS['parameter_style'].param_kwargs)

    min_voxel_size = param.Number(**PARAMS['min_voxel_size'].param_kwargs)
    noise_gain = param.Number(**PARAMS['noise_gain'].param_kwargs)

    # Sequence
    sequence_type = param.ObjectSelector(**PARAMS['sequence_type'].param_kwargs)
    pixel_bandwidth_ui = param.Selector(**PARAMS['pixel_bandwidth_ui'].param_kwargs)
    FOV_bandwidth = param.Selector(**PARAMS['FOV_bandwidth'].param_kwargs)
    FW_shift = param.Selector(**PARAMS['FW_shift'].param_kwargs)
    NSA = param.Integer(**PARAMS['NSA'].param_kwargs)
    partial_Fourier = param.Number(**PARAMS['partial_Fourier'].param_kwargs)
    turbo_factor = param.Integer(**PARAMS['turbo_factor'].param_kwargs)
    EPI_factor = param.Selector(**PARAMS['EPI_factor'].param_kwargs)

    # Contrast
    FatSat = param.Boolean(**PARAMS['FatSat'].param_kwargs)
    TR_ui = param.Selector(**PARAMS['TR_ui'].param_kwargs)
    TE_ui = param.Selector(**PARAMS['TE_ui'].param_kwargs)
    TI = param.Selector(**PARAMS['TI'].param_kwargs)
    FA = param.Selector(**PARAMS['FA'].param_kwargs)
    
    # Geometry
    trajectory = param.ObjectSelector(**PARAMS['trajectory'].param_kwargs)
    frequency_direction = param.ObjectSelector(**PARAMS['frequency_direction'].param_kwargs)
    FOV_P = param.Selector(**PARAMS['FOV_P'].param_kwargs)
    FOV_F = param.Selector(**PARAMS['FOV_F'].param_kwargs)
    phase_oversampling = param.Selector(**PARAMS['phase_oversampling'].param_kwargs)
    radial_factor = param.Number(**PARAMS['radial_factor'].param_kwargs)
    matrix_P_ui = param.Selector(**PARAMS['matrix_P_ui'].param_kwargs)
    matrix_F_ui = param.Selector(**PARAMS['matrix_F_ui'].param_kwargs)
    voxel_P = param.Selector(**PARAMS['voxel_P'].param_kwargs)
    voxel_F = param.Selector(**PARAMS['voxel_F'].param_kwargs)
    recon_matrix_P_ui = param.Selector(**PARAMS['recon_matrix_P_ui'].param_kwargs)
    recon_matrix_F_ui = param.Selector(**PARAMS['recon_matrix_F_ui'].param_kwargs)
    recon_voxel_P = param.Selector(**PARAMS['recon_voxel_P'].param_kwargs)
    recon_voxel_F = param.Selector(**PARAMS['recon_voxel_F'].param_kwargs)
    slice_thickness = param.Selector(**PARAMS['slice_thickness'].param_kwargs)
    
    radial_FOV_oversampling = param.Number(**PARAMS['radial_FOV_oversampling'].param_kwargs)
    
    # MR image
    show_FOV = param.Boolean(**PARAMS['show_FOV'].param_kwargs)
    reference_tissue = param.ObjectSelector(**PARAMS['reference_tissue'].param_kwargs)

    image_type = param.ObjectSelector(**PARAMS['image_type'].param_kwargs)

    # k-space
    show_processed_kspace = param.Boolean(**PARAMS['show_processed_kspace'].param_kwargs)
    kspace_exponent = param.Number(**PARAMS['kspace_exponent'].param_kwargs)
    kspace_type = param.ObjectSelector(**PARAMS['kspace_type'].param_kwargs)

    # Post-processing
    homodyne = param.Boolean(**PARAMS['homodyne'].param_kwargs)
    do_apodize = param.Boolean(**PARAMS['do_apodize'].param_kwargs)
    apodization_alpha = param.Number(**PARAMS['apodization_alpha'].param_kwargs)
    do_zerofill = param.Boolean(**PARAMS['do_zerofill'].param_kwargs)
    
    # Sequence plot
    shot_ui = param.Integer(**PARAMS['shot_ui'].param_kwargs)
    signal_exponent = param.Number(**PARAMS['signal_exponent'].param_kwargs)