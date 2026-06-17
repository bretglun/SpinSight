import math


def format_float(value, sigfigs=2):
    rounded = float(f'{value:.{sigfigs}g}')
    integer, decimal = str(rounded).split('.')
    exponent = int(math.floor(math.log10(abs(rounded))))
    num_decimals = sigfigs - exponent - 1
    if num_decimals <= 0:
        return integer
    decimal += '0' * (num_decimals - len(decimal))
    return '.'.join((integer, decimal))


def format_scantime(milliseconds):
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
        return f'{int(milliseconds)} msec'