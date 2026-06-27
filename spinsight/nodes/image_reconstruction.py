from spinsight import recon
from spinsight.DAG import Graph


@Graph.node()
def image_array(oversampled_recon_matrix, full_k_matrix, recon_matrix, zerofilled_kspace):
    pixel_shifts = [0., 0.]
    sample_shifts = [0., 0.]
    for dim in range(2):
        if not oversampled_recon_matrix[dim]%2:
            pixel_shifts[dim] += 1/2 # half pixel shift for even matrixsize due to fft
        if (oversampled_recon_matrix[dim] - recon_matrix[dim])%2:
            pixel_shifts[dim] += 1/2 # half pixel shift due to cropping an odd number of pixels in image space
        if not full_k_matrix[dim]%2:
            sample_shifts[dim] += 1/2 # half sample shift for even matrixsize due to fft
            if (oversampled_recon_matrix[dim] - full_k_matrix[dim])%2:
                sample_shifts[dim] -= 1 # sample shift for odd number of zeroes added
    image_array = recon.IFFT(zerofilled_kspace, pixel_shifts, sample_shifts)
    return recon.crop(image_array, recon_matrix)