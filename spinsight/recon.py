import finufft
import mrinufft
import numpy as np


def get_k_axis(matrix, pixel_size, symmetric=True, fftshift=True):
    kax = np.fft.fftfreq(matrix)
    if symmetric and not (matrix%2): # half-pixel to make even matrix symmetric
        kax += 1/(2*matrix)
    kax /= pixel_size
    if fftshift:
        kax = np.fft.fftshift(kax)
    return kax


def get_pixel_shift_matrix(matrix, shift):
    phase = [np.fft.fftfreq(matrix[dim]) * shift[dim] * 2*np.pi for dim in range(len(matrix))]
    return np.exp(1j * np.sum(np.stack(np.meshgrid(*phase[::-1])), axis=0))


def FFT(img, pixel_shifts, sample_shifts):
    half_pixel_shift = get_pixel_shift_matrix(img.shape, pixel_shifts)
    half_sample_shift = get_pixel_shift_matrix(img.shape, sample_shifts)
    kspace = np.fft.fft2(np.fft.ifftshift(img) / half_sample_shift)
    ksp = np.fft.fftshift(kspace / half_pixel_shift)
    return ksp


def IFFT(ksp, pixel_shifts, sample_shifts):
    half_pixel_shift = get_pixel_shift_matrix(ksp.shape, pixel_shifts)
    half_sample_shift = get_pixel_shift_matrix(ksp.shape, sample_shifts)
    kspace = np.fft.ifftshift(ksp) * half_pixel_shift
    img = np.fft.fftshift(np.fft.ifft2(kspace) * half_sample_shift)
    return img


def crop_or_pad(arr, shape):
    slices = [slice(None)] * arr.ndim
    pad_width = [(0, 0)] * arr.ndim

    for dim, n in enumerate(arr.shape):
        if shape[dim] < n: # crop from center
            start = (n - shape[dim]) // 2
            slices[dim] = slice(start, start + shape[dim])
        elif shape[dim] > n: # zerofill
            after = (shape[dim] - n) // 2
            before = shape[dim] - n - after
            pad_width[dim] = (before, after)

    return np.pad(arr[tuple(slices)], pad_width)


def crop(arr, shape):
    return crop_or_pad(arr, shape)


def zerofill(kspace, recon_matrix):
    return crop_or_pad(kspace, recon_matrix)


def resample_kspace_Cartesian(phantom_object, k_axes, shape=None):
    kspace = phantom_object['kspace'].copy()
    for dim in range(len(k_axes)):
        sinc = np.sinc((np.tile(k_axes[dim], (len(phantom_object['k_axes'][dim]), 1)) - np.tile(phantom_object['k_axes'][dim][:, np.newaxis], (1, len(k_axes[dim])))) * phantom_object['support'][dim])
        for tissue in kspace:
            kspace[tissue] = np.moveaxis(np.tensordot(kspace[tissue], sinc, axes=(dim, 0)), -1, dim)
    if shape is not None:
        for tissue in kspace:
            kspace[tissue] = kspace[tissue].reshape(shape)
    return kspace


def resample_kspace(phantom_object, k_samples):
    gridder = get_gridder(k_samples, phantom_object['support'])
    norm_factor = np.sqrt(4 * np.prod(gridder.shape)) # for mrinufft
    return {tissue: ungrid(phantom_object['kspace'][tissue], gridder=gridder, shape=k_samples.shape[:-1]) * norm_factor for tissue in phantom_object['kspace']}


def homodyne_weights(N, num_blank, dim):
    # create homodyne ramp filter of length N with num_blank unsampled lines
    W = np.ones((N,))
    W[num_blank:N-num_blank] = np.linspace(1, 0, N-2*(num_blank-1))[1:-1]
    W[N-num_blank:] = 0
    shape = (N, 1) if dim==0 else (1, N)
    return W.reshape(shape)


def radial_Tukey(alpha, matrix):
    kx = np.linspace(-1, 1, matrix[0]+2)[1:-1]
    ky = np.linspace(-1, 1, matrix[1]+2)[1:-1]
    kky, kkx = np.meshgrid(ky, kx)
    k = (1-np.sqrt(kkx**2 + kky**2))/alpha
    k[k>1] = 1
    k[k<0] = 0
    return np.sin(np.pi*k/2)**2


def get_gridder(k_samples, FOV):
    shape = tuple(int(np.ceil(np.max(np.abs(k_samples[..., dim])) * FOV[dim] * 2)) for dim in range(k_samples.shape[-1]))
    samples = np.array(k_samples, dtype='float32') * FOV / shape
    density = mrinufft.density.voronoi(samples)
    return mrinufft.get_operator('finufft')(2 * np.pi * samples, density=density, shape=shape)


def ungrid(gridded, k_samples=None, FOV=None, gridder=None, shape=None):
    if not gridder:
        gridder = get_gridder(k_samples, FOV)
    sample_shifts = [0 if gridded.shape[dim]%2 else 1/2 for dim in range(2)]
    img = IFFT(crop_or_pad(gridded, gridder.shape), [0, 0], sample_shifts)
    ungridded = gridder.op(img)
    if k_samples is not None:
        return ungridded.reshape(k_samples.shape[:-1])
    elif shape is not None:
        return ungridded.reshape(shape)
    return ungridded


def grid(ungridded, shape, k_samples=None, FOV=None, gridder=None):
    if (k_samples is None) == (gridder is None):
        raise ValueError('Use either k_samples or gridder, not both.')
    if gridder is None:
        gridder = get_gridder(k_samples, FOV)
    img = gridder.adj_op(ungridded.flatten())
    sample_shifts = [0 if gridder.shape[dim]%2 else 1/2 for dim in range(2)]
    gridded = FFT(img, [0, 0], sample_shifts)
    return crop_or_pad(gridded, shape)