from dataclasses import dataclass

import numpy as np
import panel as pn
import param
from panel.widgets.base import Widget

from spinsight import constants, convert, formatting, phantom, presets
from spinsight.param_utils import snap

DEFAULT_WIDGETS = {
    param.Selector: pn.widgets.Select,
    param.Number: pn.widgets.FloatSlider,
    param.Integer: pn.widgets.IntSlider,
    param.Boolean: pn.widgets.Checkbox
}


@dataclass(frozen=True)
class ParamSpec:
    label: str
    type: type[param.Parameter]
    bounds: tuple | None = None
    objects: list | dict | None = None
    step: float | None = None
    default: None = None
    precedence: int = -1
    group: str | None = None
    widget: Widget | None = None

    def __post_init__(self):
        if self.widget is None:
            object.__setattr__(self, 'widget', DEFAULT_WIDGETS[self.type])
        if self.default is not None and self.objects is not None:
            # assert default is in objects
            values = self.objects.values() if isinstance(self.objects, dict) else self.objects
            if self.default not in values:
                object.__setattr__(self, 'default', snap(self.default, values))

    @property
    def param_kwargs(self):
        param_kw = {'label', 'bounds', 'objects', 'step', 'default', 'precedence'}
        return {kw: getattr(self, kw) for kw in param_kw if getattr(self, kw) is not None}


pbw_vals = sorted({float(f'{v:.2g}') for v in np.geomspace(125, 2e3, 1000)})
fov_bw_vals = [convert.pixel_BW_to_FOV_BW(bw, 180) for bw in pbw_vals]
FW_shift_vals = [convert.pixel_BW_to_shift(pBW, 1.5) for pBW in reversed(pbw_vals)]

TR_vals = sorted({float(f'{v:.2g}') for v in np.geomspace(1, 1e4, 1000)})
TE_vals = sorted({float(f'{v:.2g}') for v in np.geomspace(1, 1e3, 1000)})
TI_vals = sorted({float(f'{v:.2g}') for v in np.geomspace(1e2, 4e3, 1000)})

fov_vals = range(100, 600 + 1)

matrix_vals = list(range(16, 600 + 1))
recon_matrix_vals = list(range(16, 1200 + 1))
voxel_vals = [240 / matrix for matrix in reversed(matrix_vals)]
recon_voxel_vals = [240 / matrix for matrix in reversed(recon_matrix_vals)]

PARAMS = {

    'load_presets': ParamSpec(
        label = 'Load preset parameters',
        type = param.Selector,
        objects = presets.get_presets(),
        default = presets.CUSTOM_PRESET,
        precedence = 1,
        group = 'Settings',
    ),

    'object': ParamSpec(
        label = 'Phantom object',
        type = param.Selector,
        objects = phantom.get_phantom_names(),
        default = 'brain',
        precedence = 2,
        group = 'Settings',
    ),

    'field_strength': ParamSpec(
        label = 'B0 field strength [T]',
        type = param.Selector,
        objects = [1.5, 3.0],
        default = 1.5,
        precedence = 3,
        group = 'Settings',
    ),

    'parameter_style': ParamSpec(
        label = 'Parameter style',
        type = param.Selector,
        objects = ['Matrix and Pixel BW', 'Voxel size and Fat/water shift', 'Matrix and FOV BW'],
        default = 'Matrix and Pixel BW',
        precedence = 4,
        group = 'Settings',
    ),

    'min_voxel_size': ParamSpec(
        label = 'Phantom resolution limit (to bound computation time) [mm]',
        type = param.Number,
        default = 0.5,
    ),
    
    'noise_gain': ParamSpec(
        label = 'Scaling factor for noise standard deviation',
        type = param.Number,
        default = 3.0,
    ),
    
    'sequence_type': ParamSpec(
        label = 'Pulse sequence',
        type = param.Selector,
        objects = ['Spin Echo', 'Spoiled Gradient Echo', 'Inversion Recovery'],
        default = 'Spin Echo',
        precedence = 1,
        group = 'Sequence',
    ),
    
    'pixel_bandwidth': ParamSpec(
        label = 'Bandwidth',
        type = param.Selector,
        objects = {formatting.pixel_bandwidth(bw): bw for bw in pbw_vals},
        default = 480.0,
        precedence = 2,
        widget = pn.widgets.DiscreteSlider,
        group = 'Sequence',
    ),

    'FOV_bandwidth': ParamSpec(
        label = 'FOV bandwidth',
        type = param.Selector,
        objects = {formatting.FOV_bandwidth(fovbw): fovbw for fovbw in fov_bw_vals},
        default = 43.2,
        precedence = -2,
        widget = pn.widgets.DiscreteSlider,
        group = 'Sequence',
    ),

    'FW_shift': ParamSpec(
        label = 'Fat/water shift',
        type = param.Selector,
        objects = {formatting.FW_shift(shift): shift for shift in FW_shift_vals},
        default = 0.45,
        precedence = -2,
        widget = pn.widgets.DiscreteSlider,
        group = 'Sequence',
    ),

    'NSA': ParamSpec(
        label = 'NSA',
        type = param.Selector,
        objects = list(range(1, 16 + 1)),
        default = 1,
        precedence = 3,
        widget = pn.widgets.DiscreteSlider,
        group = 'Sequence',
    ),

    'partial_Fourier_factor': ParamSpec(
        label = 'Partial Fourier factor',
        type = param.Selector,
        objects = {formatting.percentage(factor): float(factor) for factor in np.linspace(.6, 1, 41)},
        default = 1.00,
        precedence = 5,
        widget = pn.widgets.DiscreteSlider,
        group = 'Sequence',
    ),

    'turbo_factor': ParamSpec(
        label = 'Turbo factor',
        type = param.Selector,
        objects = list(range(1, 64 + 1)),
        default = 1,
        precedence = 6,
        widget = pn.widgets.DiscreteSlider,
        group = 'Sequence',
    ),

    'EPI_factor': ParamSpec(
        label = 'EPI factor',
        type = param.Selector,
        objects = list(range(1, 64 + 1)),
        default = 1,
        precedence = 7,
        widget = pn.widgets.DiscreteSlider,
        group = 'Sequence',
    ),

    'echo_order': ParamSpec(
        label = 'Echo order',
        type = param.Selector,
        objects = ['Linear', 'Centric', 'Flexible'],
        default = 'Flexible',
        precedence = -8,
        group = 'Sequence',
    ),

    'FatSat': ParamSpec(
        label = 'Fat saturation',
        type = param.Boolean,
        default = False,
        precedence = 1,
        group = 'Contrast',
    ),

    'TR': ParamSpec(
        label = 'TR',
        type = param.Selector,
        objects = {formatting.Ts(tr): tr for tr in TR_vals},
        default = 9000.0,
        precedence = 2,
        widget = pn.widgets.DiscreteSlider,
        group = 'Contrast',
    ),

    'TE': ParamSpec(
        label = 'TE',
        type = param.Selector,
        objects = {formatting.Ts(te): te for te in TE_vals},
        default = 12.0,
        precedence = 3,
        widget = pn.widgets.DiscreteSlider,
        group = 'Contrast',
    ),

    'TI': ParamSpec(
        label = 'TI',
        type = param.Selector,
        objects = {formatting.Ts(ti): ti for ti in TI_vals},
        default = 40.0,
        precedence = -4,
        widget = pn.widgets.DiscreteSlider,
        group = 'Contrast',
    ),

    'FA': ParamSpec(
        label = 'Flip angle',
        type = param.Selector,
        objects = {formatting.flip_angle(fa): fa for fa in range(1, 90 + 1)},
        default = 90,
        precedence = -5,
        widget = pn.widgets.DiscreteSlider,
        group = 'Contrast',
    ),

    'trajectory': ParamSpec(
        label = 'k-space trajectory',
        type = param.Selector,
        objects = {'Cartesian': 'Cartesian', 'Radial': 'Radial'},
        default = 'Cartesian',
        precedence = 1,
        group = 'Geometry',
    ),

    'frequency_direction': ParamSpec(
        label = 'Frequency encoding direction',
        type = param.Selector,
        objects = ['anterior-posterior', 'left-right'],
        default = 'anterior-posterior',
        precedence = 1,
        group = 'Geometry',
    ),

    'FOV_P': ParamSpec(
        label = 'FOV x',
        type = param.Selector,
        objects = {formatting.FOV(fov): fov for fov in fov_vals},
        default = 240,
        precedence = 2,
        widget = pn.widgets.DiscreteSlider,
        group = 'Geometry',
    ),

    'FOV_F': ParamSpec(
        label = 'FOV y',
        type = param.Selector,
        objects = {formatting.FOV(fov): fov for fov in fov_vals},
        default = 240,
        precedence = 2,
        widget = pn.widgets.DiscreteSlider,
        group = 'Geometry',
    ),

    'phase_oversampling': ParamSpec(
        label = 'Phase oversampling',
        type = param.Selector,
        objects = {formatting.percentage(factor): float(factor) for factor in np.linspace(0, 1, 101)},
        default = 0,
        precedence = 3,
        widget = pn.widgets.DiscreteSlider,
        group = 'Geometry',
    ),

    'radial_sampling_factor': ParamSpec(
        label = 'Radial sampling factor',
        type = param.Selector,
        objects = {formatting.percentage(factor): float(factor) for factor in np.linspace(0.1, 4, 391)},
        default = 1.00,
        precedence = -3,
        widget = pn.widgets.DiscreteSlider,
        group = 'Geometry',
    ),

    'matrix_P': ParamSpec(
        label = 'Acquisition matrix x',
        type = param.Selector,
        objects = matrix_vals,
        default = 180,
        precedence = 4,
        widget = pn.widgets.DiscreteSlider,
        group = 'Geometry',
    ),

    'matrix_F': ParamSpec(
        label = 'Acquisition matrix y',
        type = param.Selector,
        objects = matrix_vals,
        default = 180,
        precedence = 4,
        widget = pn.widgets.DiscreteSlider,
        group = 'Geometry',
    ),

    'voxel_P': ParamSpec(
        label = 'Voxel size x',
        type = param.Selector,
        objects =  {formatting.voxel_size(voxel): voxel for voxel in voxel_vals},
        default = 1.33, 
        precedence = -4,
        widget = pn.widgets.DiscreteSlider,
        group = 'Geometry',
    ),

    'voxel_F': ParamSpec(
        label = 'Voxel size y',
        type = param.Selector,
        objects =  {formatting.voxel_size(voxel): voxel for voxel in voxel_vals},
        default = 1.33, 
        precedence = -4,
        widget = pn.widgets.DiscreteSlider,
        group = 'Geometry',
    ),

    'recon_matrix_P': ParamSpec(
        label = 'Reconstruction matrix x',
        type = param.Selector,
        objects = recon_matrix_vals,
        default = 360,
        precedence = 5,
        widget = pn.widgets.DiscreteSlider,
        group = 'Geometry',
    ),

    'recon_matrix_F': ParamSpec(
        label = 'Reconstruction matrix y',
        type = param.Selector,
        objects = recon_matrix_vals,
        default = 360,
        precedence = 5,
        widget = pn.widgets.DiscreteSlider,
        group = 'Geometry',
    ),

    'recon_voxel_P': ParamSpec(
        label = 'Reconstructed voxel size x',
        type = param.Selector,
        objects =  {formatting.voxel_size(voxel): voxel for voxel in recon_voxel_vals},
        precedence = -5,
        widget = pn.widgets.DiscreteSlider,
        group = 'Geometry',
    ),

    'recon_voxel_F': ParamSpec(
        label = 'Reconstructed voxel size y',
        type = param.Selector,
        objects =  {formatting.voxel_size(voxel): voxel for voxel in recon_voxel_vals},
        precedence = -5,
        widget = pn.widgets.DiscreteSlider,
        group = 'Geometry',
    ),

    'slice_thickness': ParamSpec(
        label = 'Slice thickness',
        type = param.Selector,
        objects = {formatting.slice_thickness(thk): float(thk) for thk in np.linspace(0.5, 10, 96)},
        default = 3.0,
        precedence = 6,
        widget = pn.widgets.DiscreteSlider,
        group = 'Geometry',
    ),

    'radial_FOV_oversampling': ParamSpec(
        label = 'Radial FOV oversampling factor',
        type = param.Number,
        bounds = (1.0, 2.0),
        step = 0.01,
        default = 2.0,
    ),

    'show_FOV': ParamSpec(
        label = 'Show FOV',
        type = param.Boolean,
        default = False,
        precedence = 1,
    ),

    'show_TE_span': ParamSpec(
        label = 'Show TE interval',
        type = param.Boolean,
        default = False,
    ),
    
    'show_TR_span': ParamSpec(
        label = 'Show TR interval',
        type = param.Boolean,
        default = False,
    ),
    
    'show_TI_span': ParamSpec(
        label = 'Show TI interval',
        type = param.Boolean,
        default = False,
    ),

    'reference_tissue': ParamSpec(
        label = 'Reference tissue',
        type = param.Selector,
        default = 'Gray matter',
        precedence = 2,
    ),

    'image_mode': ParamSpec(
        label = 'Image mode',
        type = param.Selector,
        objects = constants.OPERATORS.keys(),
        default = 'Magnitude',
        precedence = 3,
    ),

    'show_processed_kspace': ParamSpec(
        label = 'Show processed k-space',
        type = param.Boolean,
        default = False,
        precedence = 1,
    ),

    'kspace_exponent': ParamSpec(
        label = 'k-space exponent',
        type = param.Number,
        bounds = (0.1, 1.0),
        step = 0.01,
        default = 0.20,
        precedence = 2,
    ),

    'kspace_mode': ParamSpec(
        label = 'k-space mode',
        type = param.Selector,
        objects = constants.OPERATORS.keys(),
        default = 'Magnitude',
        precedence = 3,
    ),

    'homodyne': ParamSpec(
        label = 'Homodyne',
        type = param.Boolean,
        default = True,
        precedence = -1,
        group = 'k-space processing',
    ),

    'do_apodize': ParamSpec(
        label = 'Apodization',
        type = param.Boolean,
        default = True,
        precedence = 2,
        group = 'k-space processing',
    ),

    'apodization_alpha': ParamSpec(
        label = 'Apodization alpha',
        type = param.Number,
        bounds = (.01, 1.0),
        step = 0.01,
        default = 0.25,
        precedence = 3,
        group = 'k-space processing',
    ),

    'do_zerofill': ParamSpec(
        label = 'Zerofill',
        type = param.Boolean,
        default = True,
        precedence = 4,
        group = 'k-space processing',
    ),

    'shot': ParamSpec(
        label = 'Displayed shot',
        type = param.Selector,
        objects = {f'{i + 1}': i for i in range(1000)},
        default = 0,
        precedence = 1,
        widget = pn.widgets.DiscreteSlider,
    ),

    'signal_exponent': ParamSpec(
        label = 'Signal exponent',
        type = param.Number,
        bounds = (0.1, 1.0),
        step = 0.01,
        default = 0.50,
        precedence = 2,
    ),

}