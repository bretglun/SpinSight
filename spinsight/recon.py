import finufft
import mrinufft
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


def FFT(img, pixelShifts, sampleShifts):
        halfPixelShift = getPixelShiftMatrix(img.shape, pixelShifts)
        halfSampleShift = getPixelShiftMatrix(img.shape, sampleShifts)
        kspace = np.fft.fft2(np.fft.ifftshift(img) / halfSampleShift)
        ksp = np.fft.fftshift(kspace / halfPixelShift)
        return ksp


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


def resampleKspaceCartesian(phantom, kAxes, shape=None):
    kspace = {tissue: phantom['kspace'][tissue] for tissue in phantom['kspace']}
    for dim in range(len(kAxes)):
        sinc = np.sinc((np.tile(kAxes[dim], (len(phantom['kAxes'][dim]), 1)) - np.tile(phantom['kAxes'][dim][:, np.newaxis], (1, len(kAxes[dim])))) * phantom['FOV'][dim])
        for tissue in phantom['kspace']:
            kspace[tissue] = np.moveaxis(np.tensordot(kspace[tissue], sinc, axes=(dim, 0)), -1, dim)
    if shape is not None:
        for tissue in phantom['kspace']:
            kspace[tissue] = kspace[tissue].reshape(shape)
    return kspace


def resampleKspace(phantom, kSamples):
    samples = np.array(kSamples * phantom['FOV'] / phantom['matrix'], dtype='float32')
    kspace = {}
    gridder = getGridder(samples, phantom['matrix'])
    norm_factor = np.sqrt(4 * np.prod(phantom['matrix'])) # for mrinufft
    for tissue in phantom['kspace']:
        kspace[tissue] = ungrid(phantom['kspace'][tissue], gridder=gridder, shape=samples.shape[:-1]) * norm_factor
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
        nTrailing = (reconMatrix[dim]-n)//2
        nLeading = reconMatrix[dim] - n - nTrailing
        kspace = np.insert(kspace, 0, np.zeros((nLeading, 1)), axis=dim)
        kspace = np.insert(kspace, n+nLeading, np.zeros((nTrailing, 1)), axis=dim)
    return kspace

def getKcoords(kSamples, pixelSize):
    kx = kSamples[..., 0].flatten() * 2 * np.pi * pixelSize[0]
    ky = kSamples[..., 1].flatten() * 2 * np.pi * pixelSize[1]
    return kx, ky


def getGridder(samples, shape):
    samples = np.array(samples, dtype='float32')
    density = mrinufft.density.voronoi(samples)
    return mrinufft.get_operator('finufft')(samples, density=density, shape=shape)


def ungrid(gridded, samples=None, gridder=None, shape=None):
    if not gridder:
        gridder = getGridder(samples, gridded.shape)
    sampleShifts = [0 if gridded.shape[dim]%2 else 1/2 for dim in range(2)]
    img = IFFT(gridded, [0, 0], sampleShifts)
    ungridded = gridder.op(img)
    if samples is not None:
        return ungridded.reshape(samples.shape[:-1])
    elif shape is not None:
        return ungridded.reshape(shape)
    return ungridded


def grid(ungridded, shape=None, samples=None, gridder=None):
    if not gridder:
        gridder = getGridder(samples, shape)
    img = gridder.adj_op(ungridded.flatten())
    sampleShifts = [0 if gridder.shape[dim]%2 else 1/2 for dim in range(2)]
    gridded = FFT(img, [0, 0], sampleShifts)
    return gridded


def pipeMenon2D(kx, ky, gridShape, nIter=10):
    w = np.ones(len(kx), dtype=complex)
    for iter in range(nIter):
        wGrid = grid(w, kx, ky, gridShape)
        wNonUni = ungrid(wGrid, kx, ky, w.shape)
        w /= np.abs(wNonUni)
    return w