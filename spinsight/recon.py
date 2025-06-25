import finufft
import numpy as np


def getKaxis(matrix, pixelSize, symmetric=True, fftshift=True):
    kax = np.fft.fftfreq(matrix)
    if symmetric and not (matrix%2): # half-pixel to make even matrix symmetric
        kax += 1/(2*matrix)
    kax /= pixelSize
    if fftshift:
        kax = np.fft.fftshift(kax)
    return kax


def getPixelShiftMatrix(matrix, shift):
        phase = [np.fft.fftfreq(matrix[dim]) * shift[dim] * 2*np.pi for dim in range(len(matrix))]
        return np.exp(1j * np.sum(np.stack(np.meshgrid(*phase[::-1])), axis=0))


def IFFT(ksp, pixelShifts, sampleShifts):
        halfPixelShift = getPixelShiftMatrix(ksp.shape, pixelShifts)
        halfSampleShift = getPixelShiftMatrix(ksp.shape, sampleShifts)
        kspace = np.fft.ifftshift(ksp) * halfPixelShift
        img = np.fft.fftshift(np.fft.ifft2(kspace) * halfSampleShift)
        return img


def crop(arr, shape):
    # Crop array from center according to shape
    for dim, n in enumerate(arr.shape):
        arr = arr.take(np.array(range(shape[dim])) + (n-shape[dim])//2, dim)
    return arr


def resampleKspaceCartesian(phantom, kAxes):
    kspace = {tissue: phantom['kspace'][tissue] for tissue in phantom['kspace']}
    for dim in range(len(kAxes)):
        sinc = np.sinc((np.tile(kAxes[dim], (len(phantom['kAxes'][dim]), 1)) - np.tile(phantom['kAxes'][dim][:, np.newaxis], (1, len(kAxes[dim])))) * phantom['FOV'][dim])
        for tissue in phantom['kspace']:
            kspace[tissue] = np.moveaxis(np.tensordot(kspace[tissue], sinc, axes=(dim, 0)), -1, dim)
    return kspace


def resampleKspace(phantom, kSamples):
    kspace = {}
    kx, ky = getKcoords(kSamples, [phantom['FOV'][d]/phantom['matrix'][d] for d in range(2)])
    for tissue in phantom['kspace']:
        kspace[tissue] = ungrid(phantom['kspace'][tissue], kx, ky, kSamples.shape[:-1])
    return kspace


def homodyneWeights(N, nBlank, dim):
    # create homodyne ramp filter of length N with nBlank unsampled lines
    W = np.ones((N,))
    W[nBlank-1:-nBlank+1] = np.linspace(1,0, N-2*(nBlank-1))
    W[-nBlank:] = 0
    shape = (N, 1) if dim==0 else (1, N)
    return W.reshape(shape)


def radialTukey(alpha, matrix):
    kx = np.linspace(-1, 1, matrix[0]+2)[1:-1]
    ky = np.linspace(-1, 1, matrix[1]+2)[1:-1]
    kky, kkx = np.meshgrid(ky, kx)
    k = (1-np.sqrt(kkx**2 + kky**2))/alpha
    k[k>1] = 1
    k[k<0] = 0
    return np.sin(np.pi*k/2)**2


def zerofill(kspace, reconMatrix):
    for dim, n in enumerate(kspace.shape):
        nLeading = reconMatrix[dim]//2 - n//2
        nTrailing = reconMatrix[dim] - n - nLeading
        kspace = np.insert(kspace, 0, np.zeros((nLeading, 1)), axis=dim)
        kspace = np.insert(kspace, n+nLeading, np.zeros((nTrailing, 1)), axis=dim)
    return kspace

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