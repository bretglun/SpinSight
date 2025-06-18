import finufft
import numpy as np


def getKcoords(kSamples, pixelSize):
    kx = kSamples[..., 0].flatten() * 2 * np.pi * pixelSize[0]
    ky = kSamples[..., 1].flatten() * 2 * np.pi * pixelSize[1]
    return kx, ky


def ungrid(gridded, kx, ky, shape):
    img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(gridded)))
    ungridded = finufft.nufft2d2(kx, ky, img).reshape(shape)
    return ungridded


def grid(ungridded, kx, ky, shape):
    img = finufft.nufft2d1(kx, ky, ungridded, shape)
    gridded = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))
    return gridded


def pipeMenon2D(kx, ky, gridShape, nIter=10):
    w = np.ones(len(kx), dtype=complex)
    for iter in range(nIter):
        wGrid = grid(w, kx, ky, gridShape)
        wNonUni = ungrid(wGrid, kx, ky, w.shape)
        w /= np.abs(wNonUni)
    return w