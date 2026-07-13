from spinsight import recon
from spinsight.DAG import Graph


@Graph.node()
def image_array(oversampled_image_array, recon_matrix):
    return recon.crop(oversampled_image_array, recon_matrix)


@Graph.node()
def oversampled_image_array(zerofilled_kspace, pixel_shifts, sample_shifts):
    return recon.IFFT(zerofilled_kspace, pixel_shifts, sample_shifts)


@Graph.node()
def pixel_shifts(oversampled_recon_matrix, recon_matrix):
    pixel_shifts = [0., 0.]
    for dim in range(2):
        if not oversampled_recon_matrix[dim]%2:
            pixel_shifts[dim] += 1/2 # half pixel shift for even matrixsize due to fft
        if (oversampled_recon_matrix[dim] - recon_matrix[dim])%2:
            pixel_shifts[dim] += 1/2 # half pixel shift due to cropping an odd number of pixels in image space
    return pixel_shifts


@Graph.node()
def sample_shifts(oversampled_recon_matrix, full_k_matrix):
    sample_shifts = [0., 0.]
    for dim in range(2):
        if not full_k_matrix[dim]%2:
            sample_shifts[dim] += 1/2 # half sample shift for even matrixsize due to fft
            if (oversampled_recon_matrix[dim] - full_k_matrix[dim])%2:
                sample_shifts[dim] -= 1 # sample shift for odd number of zeroes added
    return sample_shifts