from dataclasses import dataclass
from spinsight import constants, convert, formatting, phantom
import numpy as np


def snap(value, values, mode='nearest'):
    match mode:
        case 'nearest':
            return min(values, key=lambda x: abs(x-value), default=None)
        case 'ceil':
            return min([v for v in values if v >= value], default=None)
        case 'floor':
            return max([v for v in values if v <= value], default=None)
        case _:
            raise ValueError(f'Invalid mode {mode}')


@dataclass(frozen=True)
class ParamSpec:
    label: str
    bounds: tuple = None
    objects: list | dict = None
    step: float = None
    default: None = None
    precedence: int = -1

    derived: bool = False

    def __post_init__(self):
        if self.default is not None and self.objects is not None:
            # assert default is in objects
            values = self.objects.values() if isinstance(self.objects, dict) else self.objects
            if self.default not in values:
                object.__setattr__(self, 'default', snap(self.default, values))

    @property
    def param_kwargs(self):
        param_kw = {'label', 'bounds', 'objects', 'step', 'default', 'precedence'}
        return {kw: getattr(self, kw) for kw in param_kw if getattr(self, kw) is not None}


pbw_vals = [float(v) for v in np.geomspace(125, 2e3, 500)]
fov_bw_vals = [convert.pixel_BW_to_FOV_BW(bw, 180) for bw in pbw_vals]
FW_shift_vals = [convert.pixel_BW_to_shift(pBW, 1.5) for pBW in pbw_vals[::-1]]

TR_vals = [float(v) for v in np.geomspace(1, 1e4, 500)]
TE_vals = [float(v) for v in np.geomspace(1, 1e3, 500)]
TI_vals = [float(v) for v in np.geomspace(1e2, 4e3, 500)]

fov_vals = range(100, 600 + 1)

matrix_vals = list(range(16, 600 + 1))
recon_matrix_vals = list(range(16, 1200 + 1))
voxel_vals = [240 / matrix for matrix in matrix_vals[::-1]]
recon_voxel_vals = [240 / matrix for matrix in recon_matrix_vals[::-1]]

PARAMS = {

    'object': ParamSpec(
        label = 'Phantom object',
        objects = phantom.get_phantom_names(),
        default = 'brain',
        precedence = 1,
    ),

    'field_strength': ParamSpec(
        label = 'B0 field strength [T]',
        objects = [1.5, 3.0],
        default = 1.5,
        precedence = 2,
    ),

    'parameter_style': ParamSpec(
        label = 'Parameter Style',
        objects = ['Matrix and Pixel BW', 'Voxel size and Fat/water shift', 'Matrix and FOV BW'],
        default = 'Matrix and Pixel BW',
        precedence = 3,
    ),

    'min_voxel_size': ParamSpec(
        label = 'Phantom resolution limit (to bound computation time) [mm]',
        default = 0.5,
    ),
    
    'noise_gain': ParamSpec(
        label = 'Scaling factor for noise standard deviation',
        default = 3.0,
    ),
    
    'sequence_type': ParamSpec(
        label = 'Pulse sequence',
        objects = ['Spin Echo', 'Spoiled Gradient Echo', 'Inversion Recovery'],
        default = 'Spin Echo',
        precedence = 1,
    ),
    
    'pixel_bandwidth_ui': ParamSpec(
        label = 'Pixel bandwidth',
        objects = {formatting.pixel_bandwidth(bw): bw for bw in pbw_vals},
        default = 480,
        precedence = 2,
    ),

    'FOV_bandwidth': ParamSpec(
        label = 'FOV bandwidth',
        objects = {formatting.FOV_bandwidth(fovbw): fovbw for fovbw in fov_bw_vals},
        default = 43.2,
        precedence = 2,
        derived = True,
    ),

    'FW_shift': ParamSpec(
        label = 'Fat/water shift',
        objects = {formatting.FW_shift(shift): shift for shift in FW_shift_vals},
        default = 0.45,
        precedence = 2,
        derived = True,
    ),

    'NSA': ParamSpec(
        label = 'NSA',
        bounds = (1, 16),
        default = 1,
        precedence = 3,
    ),

    'partial_Fourier': ParamSpec(
        label = 'Partial Fourier factor',
        bounds = (.6, 1),
        step = 0.01,
        default = 1,
        precedence = 5,
    ),

    'turbo_factor': ParamSpec(
        label = 'Turbo factor',
        bounds = (1, 64),
        default = 1,
        precedence = 6,
    ),

    'EPI_factor': ParamSpec(
        label = 'EPI factor',
        objects = list(range(1, 64 + 1)),
        default = 1,
        precedence = 7,
    ),

    'FatSat': ParamSpec(
        label = 'Fat saturation',
        default = False,
        precedence = 1,
    ),

    'TR_ui': ParamSpec(
        label = 'TR',
        objects = {formatting.Ts(tr): tr for tr in TR_vals},
        default = 9000,
        precedence = 2,
    ),

    'TE_ui': ParamSpec(
        label = 'TE',
        objects = {formatting.Ts(te): te for te in TE_vals},
        default = 12,
        precedence = 3,
    ),

    'TI': ParamSpec(
        label = 'TI',
        objects = {formatting.Ts(ti): ti for ti in TI_vals},
        default = 40,
        precedence = 4,
    ),

    'FA': ParamSpec(
        label = 'Flip angle',
        objects = {formatting.flip_angle(fa): fa for fa in range(1, 90 + 1)},
        default = 90,
        precedence = 5,
    ),

    'trajectory': ParamSpec(
        label = 'k-space trajectory',
        objects = ['Cartesian', 'Radial', 'PROPELLER'],
        default = 'Cartesian',
        precedence = 1,
    ),

    'frequency_direction': ParamSpec(
        label = 'Frequency encoding direction',
        objects = ['anterior-posterior', 'left-right'],
        default = 'anterior-posterior',
        precedence = 1,
    ),

    'FOV_P': ParamSpec(
        label = 'FOV x',
        objects = {formatting.FOV(fov): fov for fov in fov_vals},
        default = 240,
        precedence = 2,
    ),

    'FOV_F': ParamSpec(
        label = 'FOV y',
        objects = {formatting.FOV(fov): fov for fov in fov_vals},
        default = 240,
        precedence = 2,
    ),

    'phase_oversampling': ParamSpec(
        label = 'Phase oversampling',
        objects = {formatting.phase_oversampling(factor): float(factor) for factor in np.linspace(1, 2, 101)},
        default = 0,
        precedence = 3,
    ),

    'radial_factor': ParamSpec(
        label = 'Spoke sampling factor',
        bounds = (0.1, 4.0),
        default = 1.0,
        precedence = 3,
    ),

    'matrix_P_ui': ParamSpec(
        label = 'Acquisition matrix x',
        objects = matrix_vals,
        default = 180,
        precedence = 4,
    ),

    'matrix_F_ui': ParamSpec(
        label = 'Acquisition matrix y',
        objects = matrix_vals,
        default = 180,
        precedence = 4,
    ),

    'voxel_P': ParamSpec(
        label = 'Voxel size x',
        objects =  {formatting.voxel_size(voxel): voxel for voxel in voxel_vals},
        default = 1.333, 
        precedence = 4,
        derived = True,
    ),

    'voxel_F': ParamSpec(
        label = 'Voxel size y',
        objects =  {formatting.voxel_size(voxel): voxel for voxel in voxel_vals},
        default = 1.333, 
        precedence = 4,
        derived = True,
    ),

    'recon_matrix_P_ui': ParamSpec(
        label = 'Reconstruction matrix x',
        objects = recon_matrix_vals,
        default = 360,
        precedence = 5,
    ),

    'recon_matrix_F_ui': ParamSpec(
        label = 'Reconstruction matrix y',
        objects = recon_matrix_vals,
        default = 360,
        precedence = 5,
    ),

    'recon_voxel_P': ParamSpec(
        label = 'Reconstructed voxel size x',
        objects =  {formatting.voxel_size(voxel): voxel for voxel in recon_voxel_vals},
        precedence = 5,
        derived = True,
    ),

    'recon_voxel_F': ParamSpec(
        label = 'Reconstructed voxel size y',
        objects =  {formatting.voxel_size(voxel): voxel for voxel in recon_voxel_vals},
        precedence = 5,
        derived = True,
    ),

    'slice_thickness': ParamSpec(
        label = 'Slice thickness',
        objects = {formatting.slice_thickness(thk): thk for thk in np.linspace(0.5, 10, 96)},
        default = 3.0,
        precedence = 6,
    ),

    'radial_FOV_oversampling': ParamSpec(
        label = 'Radial FOV oversampling factor',
        bounds = (1.0, 2.0),
        step = 0.01,
        default = 2.0,
    ),

    'rec_acq_ratio_P': ParamSpec(
        label = 'Reconstructed / acquired matrix_P ratio',
        default = 2.0,
        derived = True,
    ),

    'rec_acq_ratio_F': ParamSpec(
        label = 'Reconstructed / acquired matrix_F ratio',
        default = 2.0,
        derived = True,
    ),

    'show_FOV': ParamSpec(
        label = 'Show FOV',
        default = False,
        precedence = 1,
    ),

    'reference_tissue': ParamSpec(
        label = 'Reference tissue',
        precedence = 2,
    ),

    'image_type': ParamSpec(
        label = 'Image type',
        objects = constants.OPERATORS.keys(),
        default = 'Magnitude',
    ),

    'show_processed_kspace': ParamSpec(
        label = 'Show processed k-space',
        default = False,
        precedence = 1,
    ),

    'kspace_exponent': ParamSpec(
        label = 'k-space exponent',
        bounds = (0.1, 1.0),
        step = 0.01,
        default = 0.2,
        precedence = 2,
    ),

    'kspace_type': ParamSpec(
        label = 'k-space type',
        objects = constants.OPERATORS.keys(),
        default = 'Magnitude',
    ),

    'homodyne': ParamSpec(
        label = 'Homodyne',
        default = True,
        precedence = 1,
    ),

    'do_apodize': ParamSpec(
        label = 'Apodization',
        default = True,
        precedence = 2,
    ),

    'apodization_alpha': ParamSpec(
        label = 'Apodization alpha',
        bounds = (.01, 1.0),
        step = 0.01,
        default = 0.25,
        precedence = 3,
    ),

    'do_zerofill': ParamSpec(
        label = 'Zerofill',
        default = True,
        precedence = 4,
    ),

    'shot_ui': ParamSpec(
        label = 'Displayed shot',
        default = 1,
        precedence = 1,
    ),

    'signal_exponent': ParamSpec(
        label = 'Signal exponent',
        bounds = (0.1, 1.0),
        step = 0.01,
        default = 0.5,
        precedence = 2,
    ),

}