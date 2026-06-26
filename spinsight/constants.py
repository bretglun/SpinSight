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

OPERATORS = {'Magnitude': np.abs, 'Phase': np.angle, 'Real': np.real, 'Imaginary': np.imag}

MAX_PHASE_OVERSAMPLING_FACTOR = 2

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