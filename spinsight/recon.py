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


def crop(arr, shape):
    # Crop array from center according to shape
    for dim, n in enumerate(arr.shape):
        arr = arr.take(np.array(range(shape[dim])) + (n-shape[dim])//2, dim)
    return arr


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
    samples = np.array(k_samples * phantom_object['support'] / phantom_object['matrix'], dtype='float32')
    kspace = {}
    gridder = get_gridder(samples, phantom_object['matrix'])
    norm_factor = np.sqrt(4 * np.prod(phantom_object['matrix'])) # for mrinufft
    for tissue in phantom_object['kspace']:
        kspace[tissue] = ungrid(phantom_object['kspace'][tissue], gridder=gridder, shape=samples.shape[:-1]) * norm_factor
    return kspace


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


def zerofill(kspace, recon_matrix):
    for dim, n in enumerate(kspace.shape):
        num_trailing = (recon_matrix[dim]-n)//2
        num_leading = recon_matrix[dim] - n - num_trailing
        kspace = np.insert(kspace, 0, np.zeros((num_leading, 1)), axis=dim)
        kspace = np.insert(kspace, n+num_leading, np.zeros((num_trailing, 1)), axis=dim)
    return kspace

def get_k_coords(k_samples, pixel_size):
    kx = k_samples[..., 0].flatten() * 2 * np.pi * pixel_size[0]
    ky = k_samples[..., 1].flatten() * 2 * np.pi * pixel_size[1]
    return kx, ky


def get_gridder(samples, shape):
    for dim in range(len(shape)):
        if np.max(np.abs(samples[..., dim])) > .5:
            # pad matrix to ensure samples are <= .5
            N = int(np.ceil(np.max(np.abs(samples[..., dim])) * 2 * shape[dim]))
            samples[..., dim] *= shape[dim] / N
            shape = tuple(N if d==dim else n for d, n in enumerate(shape))
    samples = np.array(samples, dtype='float32')
    density = mrinufft.density.voronoi(samples)
    return mrinufft.get_operator('finufft')(samples, density=density, shape=shape)


def ungrid(gridded, samples=None, gridder=None, shape=None):
    if not gridder:
        gridder = get_gridder(samples, gridded.shape)
    sample_shifts = [0 if gridded.shape[dim]%2 else 1/2 for dim in range(2)]
    img = IFFT(gridded, [0, 0], sample_shifts)
    ungridded = gridder.op(img)
    if samples is not None:
        return ungridded.reshape(samples.shape[:-1])
    elif shape is not None:
        return ungridded.reshape(shape)
    return ungridded


def grid(ungridded, shape, samples=None, gridder=None):
    if (samples is None) == (gridder is None):
        raise ValueError('Use either samples or gridder, not both.')
    if gridder is None:
        gridder = get_gridder(samples, shape)
    img = gridder.adj_op(ungridded.flatten())
    sample_shifts = [0 if gridder.shape[dim]%2 else 1/2 for dim in range(2)]
    gridded = FFT(img, [0, 0], sample_shifts)
    return crop(gridded, shape) # crop in case gridder shape was padded


def Pipe_Menon_2D(kx, ky, grid_shape, num_iter=10):
    w = np.ones(len(kx), dtype=complex)
    for iter in range(num_iter):
        w_grid = grid(w, kx, ky, grid_shape)
        w_nonuniform = ungrid(w_grid, kx, ky, w.shape)
        w /= np.abs(w_nonuniform)
    return w