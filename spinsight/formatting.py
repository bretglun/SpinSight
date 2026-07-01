import numpy as np


def format_float(value, sigfigs=2):
    rounded = float(f'{value:.{sigfigs}g}')
    integer, decimal = str(rounded).split('.')
    exponent = int(np.floor(np.log10(abs(rounded))))
    num_decimals = sigfigs - exponent - 1
    if num_decimals <= 0:
        return integer
    decimal += '0' * (num_decimals - len(decimal))
    return '.'.join((integer, decimal))


def scantime(milliseconds):
    total_seconds = milliseconds / 1000
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)

    if minutes > 0:
        return f'{minutes} min {seconds} sec'
    elif seconds >= 10:
        return f'{seconds} sec'
    elif seconds > 0:
        return f'{total_seconds:.1f} sec'
    else:
        return f'{milliseconds:.0f} msec'


def pixel_bandwidth(bw):
    return f'{format_float(bw, 3)} Hz / pixel'


def FOV_bandwidth(fov_bw):
    return f'±{format_float(fov_bw, 3)} kHz'


def FW_shift(shift):
    return f'{format_float(shift, 3)} pixels'


def Ts(t):
    return f'{format_float(t, 2)} msec'


def flip_angle(fa):
    return f'{fa:.0f}°'


def spoke_angle(angle):
    return f'{angle:.0f}°'


def FOV(fov):
    return f'{fov:.0f} mm'


def relative_SNR(snr):
    return f'{snr * 100:.0f}%'


def phase_oversampling(factor):
    return f'{factor * 100:.0f}%'


def voxel_size(voxel):
    return f'{format_float(voxel, 3)} mm'


def slice_thickness(thk):
    return f'{format_float(thk, 2)} mm'