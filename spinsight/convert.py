from spinsight import constants
import numpy as np


def pixel_BW_to_shift(pixel_BW, B0=1.5):
    ''' Get fat/water chemical shift [pixels] from pixel bandwidth [Hz/pixel] and B0 [T]'''
    return np.abs(constants.FAT_RESONANCES['Fat2']['shift'] * constants.GYRO * B0 / pixel_BW)


def shift_to_pixel_BW(shift, B0=1.5):
    ''' Get pixel bandwidth [Hz/pixel] from fat/water chemical shift [pixels] and B0 [T]'''
    return np.abs(constants.FAT_RESONANCES['Fat2']['shift'] * constants.GYRO * B0 / shift)


def pixel_BW_to_FOV_BW(pixel_BW, matrix_F):
    ''' Get FOV bandwidth [±kHz] from pixel bandwidth [Hz/pixel] and read direction matrix'''
    return pixel_BW * matrix_F / 2e3


def FOV_BW_to_pixel_BW(FOV_BW, matrix_F):
    ''' Get pixel bandwidth [Hz/pixel] from FOV bandwidth [±kHz] and read direction matrix'''
    return FOV_BW / matrix_F * 2e3