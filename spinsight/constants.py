import numpy as np
from spinsight import convert, formatting


GYRO = 42.577 # 1H gyromagnetic ratio [MHz/T]

TISSUES = {
    'Gray matter':                 {'PD': 1.0  , 'FF': .00, 'T1': {1.5: 1100, 3.0: 1330}, 'T2': {1.5:   95, 3.0:   99}},
    'White matter':                {'PD': 0.9  , 'FF': .00, 'T1': {1.5:  560, 3.0:  830}, 'T2': {1.5:   72, 3.0:   69}},
    'CSF':                         {'PD': 1.0  , 'FF': .00, 'T1': {1.5: 4280, 3.0: 4160}, 'T2': {1.5: 2030, 3.0: 2100}},
    'Adipose tissue':              {'PD': 1.0  , 'FF': .95, 'T1': {1.5:  290, 3.0:  370}, 'T2': {1.5:  165, 3.0:  130}},
    'Bone marrow':                 {'PD': 1.0  , 'FF': .50, 'T1': {1.5:  856, 3.0:  898}, 'T2': {1.5:   46, 3.0:   34}}, # relaxation times for water component
    'Liver':                       {'PD': 1.0  , 'FF': .01, 'T1': {1.5:  586, 3.0:  809}, 'T2': {1.5:   46, 3.0:   34}},
    'Spleen':                      {'PD': 1.0  , 'FF': .00, 'T1': {1.5: 1057, 3.0: 1328}, 'T2': {1.5:   79, 3.0:   61}},
    'Muscle':                      {'PD': 1.0  , 'FF': .00, 'T1': {1.5:  856, 3.0:  898}, 'T2': {1.5:   27, 3.0:   29}},
    'Kidney medulla':              {'PD': 1.0  , 'FF': .00, 'T1': {1.5: 1412, 3.0: 1545}, 'T2': {1.5:   85, 3.0:   81}},
    'Kidney cortex':               {'PD': 1.0  , 'FF': .00, 'T1': {1.5:  966, 3.0: 1142}, 'T2': {1.5:   87, 3.0:   76}},
    'Spinal cord':                 {'PD': 1.0  , 'FF': .00, 'T1': {1.5:  745, 3.0:  993}, 'T2': {1.5:   74, 3.0:   78}},
    'Cortical bone':               {'PD': 0.05 , 'FF': .00, 'T1': {1.5: 1000, 3.0: 1000}, 'T2': {1.5:    3, 3.0:    1}},
    'Blood':                       {'PD': 1.0  , 'FF': .00, 'T1': {1.5: 1441, 3.0: 1932}, 'T2': {1.5:  290, 3.0:  275}},
    'Stomach':                     {'PD': 1.0  , 'FF': .00, 'T1': {1.5: 3000, 3.0: 3000}, 'T2': {1.5:  800, 3.0:  800}},
    'Perotineum':                  {'PD': 1.0  , 'FF': .00, 'T1': {1.5: 1500, 3.0: 1500}, 'T2': {1.5:   30, 3.0:   30}},
    'Shepp-Logan Scalp':           {'PD': 0.8  , 'FF': .00, 'T1': {1.5:  343, 3.0:  377}, 'T2': {1.5:   70, 3.0:   70}},
    'Shepp-Logan Bone and Marrow': {'PD': 0.12 , 'FF': .00, 'T1': {1.5:  552, 3.0:  587}, 'T2': {1.5:   50, 3.0:   50}},
    'Shepp-Logan CSF':             {'PD': 0.98 , 'FF': .00, 'T1': {1.5: 4200, 3.0: 4200}, 'T2': {1.5: 1990, 3.0: 1990}},
    'Shepp-Logan Gray matter':     {'PD': 0.745, 'FF': .00, 'T1': {1.5:  998, 3.0: 1295}, 'T2': {1.5:  100, 3.0:  100}},
    'Shepp-Logan White matter':    {'PD': 0.617, 'FF': .00, 'T1': {1.5:  681, 3.0:  887}, 'T2': {1.5:   80, 3.0:   80}},
    'Shepp-Logan Tumor':           {'PD': 0.95 , 'FF': .00, 'T1': {1.5: 1011, 3.0: 1175}, 'T2': {1.5:  100, 3.0:  100}},
}
# SL = Shepp-Logan phantom tissues as defined in:
# Gach, H., Costin Tanase, and Fernando E. Boada. 2008. “2D & 3D Shepp-Logan Phantom Standards for MRI.” In: Int Conf Syst Eng, 521–26.


FAT_RESONANCES = {'Fat1': {'shift': 0.9 - 4.7, 'ratio': .087, 'ratio_with_FatSat': .010, 'PD': 1.0, 'T1': {1.5:  290, 3.0:  370}, 'T2': {1.5:  165, 3.0:  130}},
                  'Fat2': {'shift': 1.3 - 4.7, 'ratio': .694, 'ratio_with_FatSat': .033, 'PD': 1.0, 'T1': {1.5:  290, 3.0:  370}, 'T2': {1.5:  165, 3.0:  130}},
                  'Fat3': {'shift': 2.1 - 4.7, 'ratio': .129, 'ratio_with_FatSat': .038, 'PD': 1.0, 'T1': {1.5:  290, 3.0:  370}, 'T2': {1.5:  165, 3.0:  130}},
                  'Fat4': {'shift': 2.8 - 4.7, 'ratio': .004, 'ratio_with_FatSat': .003, 'PD': 1.0, 'T1': {1.5:  290, 3.0:  370}, 'T2': {1.5:  165, 3.0:  130}},
                  'Fat5': {'shift': 4.3 - 4.7, 'ratio': .039, 'ratio_with_FatSat': .037, 'PD': 1.0, 'T1': {1.5:  290, 3.0:  370}, 'T2': {1.5:  165, 3.0:  130}}, 
                  'Fat6': {'shift': 5.3 - 4.7, 'ratio': .047, 'ratio_with_FatSat': .045, 'PD': 1.0, 'T1': {1.5:  290, 3.0:  370}, 'T2': {1.5:  165, 3.0:  130}} }

SEQUENCES = ['Spin Echo', 'Spoiled Gradient Echo', 'Inversion Recovery']

DIRECTIONS = {'anterior-posterior': 0, 'left-right': 1}

TRAJECTORIES = ['Cartesian', 'Radial', 'PROPELLER']

OPERATORS = {'Magnitude': np.abs, 'Phase': np.angle, 'Real': np.real, 'Imaginary': np.imag}

PARAMETER_STYLES = ['Matrix and Pixel BW', 'Voxel size and Fat/water shift', 'Matrix and FOV BW']

bw_vals = np.exp(np.linspace(np.log(125), np.log(2000), 500))

MIN_MATRIX = 16
MAX_MATRIX = 600
MIN_RECON_MATRIX = 16
MAX_RECON_MATRIX = 1200

PARAM_VALUES = {
    'TR': {tr + ' msec': float(tr) for tr in [formatting.format_float(tr, 2) for tr in 10.**np.linspace(0, 4, 500)]},
    'TE': {te + ' msec': float(te) for te in [formatting.format_float(te, 2) for te in 10.**np.linspace(0, 3, 500)]},
    'FA': {str(int(fa)) + '°': float(fa) for fa in range(1, 91)},
    'TI': {ti + ' msec': float(ti) for ti in [formatting.format_float(ti, 2) for ti in 10.**np.linspace(1.6, 4, 500)]},
    'FOV_P': {str(int(fov)) + ' mm': float(fov) for fov in range(100, 601)},
    'FOV_F': {str(int(fov)) + ' mm': float(fov) for fov in range(100, 601)},
    'phase_oversampling': {str(int(po)) + '%': float(po) for po in range(0, 101)},
    'pixel_bandwidth_param': {bw + ' Hz': float(bw) for bw in [formatting.format_float(bw, 3) for bw in bw_vals]},
    'FOV_bandwidth': {f'±{formatting.format_float(fovbw, 3)} kHz': fovbw for fovbw in [convert.pixel_BW_to_FOV_BW(bw, 180) for bw in bw_vals]},
    'FW_shift': {f'{formatting.format_float(shift, 3)} pixels': shift for shift in [convert.pixel_BW_to_shift(pBW, 1.5) for pBW in bw_vals[::-1]]},
    'matrix_P_param': list(range(MIN_MATRIX, MAX_MATRIX+1)),
    'matrix_F_param': list(range(MIN_MATRIX, MAX_MATRIX+1)),
    'voxel_P': {f'{formatting.format_float(voxel, 3)} mm': voxel for voxel in [240 / matrix for matrix in list(range(MIN_MATRIX, MAX_MATRIX+1))[::-1]]},
    'voxel_F': {f'{formatting.format_float(voxel, 3)} mm': voxel for voxel in [240 / matrix for matrix in list(range(MIN_MATRIX, MAX_MATRIX+1))[::-1]]},
    'recon_matrix_P_param': list(range(MIN_RECON_MATRIX, MAX_RECON_MATRIX+1)),
    'recon_matrix_F_param': list(range(MIN_RECON_MATRIX, MAX_RECON_MATRIX+1)),
    'recon_voxel_P': {f'{formatting.format_float(voxel, 3)} mm': voxel for voxel in [240 / matrix for matrix in list(range(MIN_RECON_MATRIX, MAX_RECON_MATRIX+1))[::-1]]},
    'recon_voxel_F': {f'{formatting.format_float(voxel, 3)} mm': voxel for voxel in [240 / matrix for matrix in list(range(MIN_RECON_MATRIX, MAX_RECON_MATRIX+1))[::-1]]},
    'slice_thickness': {thk + ' mm': float(thk) for thk in [formatting.format_float(thk, 2) for thk in np.linspace(0.5, 10, 96)]},
    'EPI_factor': list(range(1, 64+1)),
}

MAX_AMP = 25. # mT/m
MAX_SLEW = 80. # T/m/s
INVERSION_THK_FACTOR = 1.1 # make inversion slice 10% thicker

BOARD_COLORS = {
    'frequency': 'cadetblue',
    'phase': 'cadetblue',
    'slice': 'cadetblue',
    'RF': 'red',
    'signal': 'orange',
    'ADC': 'peru',
}