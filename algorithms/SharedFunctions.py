import numpy as np
import math
import cv2
import numba as nb
import bm3d


@nb.vectorize(['float32(float32, float32, float32, float32, float32, float32)'], target='cpu')
def bilinear_interpolation(x, y, top_left, top_right, bottom_left, bottom_right):
    """
    Perform Bilinear Interpolation to find the greyscale value of a point within a pixel using its surrounding pixels.
    Adapted from https://github.com/scikit-image/scikit-image/blob/master/skimage/_shared/interpolation.pxd
    Fastest implementation.
    """
    dx = x - math.floor(x)
    dy = y - math.floor(y)

    top = (1 - dy) * top_left + dy * top_right
    bottom = (1 - dy) * bottom_left + dy * bottom_right
    return (1 - dx) * top + dx * bottom


@nb.vectorize(['float32(float32, float32, float32, float32, float32, float32)'], target='cuda')
def bilinear_interpolation_gpu(x, y, top_left, top_right, bottom_left, bottom_right):
    """
    Perform Bilinear Interpolation to find the greyscale value of a point within a pixel using its surrounding pixels.
    Adapted from https://github.com/scikit-image/scikit-image/blob/master/skimage/_shared/interpolation.pxd
    Utilises cuda.
    Note: In tests this was found to be most slow, likely due to GPU copying overheads
    """
    dx = x - math.floor(x)
    dy = y - math.floor(y)

    top = (1 - dy) * top_left + dy * top_right
    bottom = (1 - dy) * bottom_left + dy * bottom_right
    return (1 - dx) * top + dx * bottom



@nb.vectorize(['float32(float32, float32, float32, float32, float32, float32)'], target='parallel')
def bilinear_interpolation_parallel(x, y, top_left, top_right, bottom_left, bottom_right):
    """
    Perform Bilinear Interpolation to find the greyscale value of a point within a pixel using its surrounding pixels.
    Adapted from https://github.com/scikit-image/scikit-image/blob/master/skimage/_shared/interpolation.pxd
    Note: In tests this was slower than single-CPU in bilinear_interpolation
    """
    dx = x - math.floor(x)
    dy = y - math.floor(y)

    top = (1 - dy) * top_left + dy * top_right
    bottom = (1 - dy) * bottom_left + dy * bottom_right
    return (1 - dx) * top + dx * bottom


def get_riu2_mappings(p: int):
    """
    Generate a mapping table from an LBP value to a riu2 bin.

    To summarise, riu2 classifies uniform LBPs (defined as U<=2, where U is the number of bitwise transitions) into p+1
    groups, and then puts non-uniform patterns into one group.
    """
    # Number of distinct LBP patterns to create mappings for
    size = 2 ** p

    # Create riu2 mappings table of unsigned integers
    mappings = [0] * size

    for i in range(0, size):
        # Convert i to representation as binary string
        binary = format(i, '08b')
        # Bit shift left by taking first digit and concatenating to end
        binary_lshift = binary[1:] + binary[:1]

        # Effectively, take each digit of 'binary' and 'binary_lshift' and check if there is a value difference.
        # When this is the case, it indicates a bitwise transition from 0 to 1 or 1 to 0.
        U, sum_bits = 0, 0
        for j in range(0, p):
            bit = binary[j]
            bit_lshift = binary_lshift[j]

            if bit != bit_lshift:
                # bitwise transition
                U += 1
            sum_bits += int(bit)  # Keep sum of bits

        if U <= 2:
            # Put uniform patterns into one of p+1 groups
            # Eg: If p=4 we have a group for every sum of bits. 0000 = 0, 0001 = 1, 0011 = 2, 0111 = 3, 1111 = 4
            # Forming 5 separate groups
            mappings[i] = sum_bits
        else:
            # Put non-uniform patterns into one group
            mappings[i] = p + 1  # p + 1 case

    return mappings


@nb.jit('void(float32[:,:], float32[:], float32[:], uint16, float32[:])', nopython=True)
def get_radial_means(image, x_pos, y_pos, patch_offset, means):
    """
    Gets the mean of a w*w size patch centred on x_pos and y_pos
    """
    # Calculate radial means for each position in r
    for i in range(len(x_pos)):
        # No interpolation required
        if np.floor(x_pos[i]) == x_pos[i] and np.floor(y_pos[i]) == y_pos[i]:
            x, y = int(x_pos[i]), int(y_pos[i])
            if patch_offset == 0:
                # If no patch_offset, no mean is required.
                means[i] = image[x, y]
            else:
                neighbour_patch = image[x - patch_offset:x + patch_offset + 1, y - patch_offset:y + patch_offset + 1]
                means[i] = np.mean(neighbour_patch)
        else:
            # Interpolation required
            minx, miny = math.floor(x_pos[i]), math.floor(y_pos[i])
            dx, dy = x_pos[i] - minx, y_pos[i] - miny

            # Find mean of w_r1*1_r1 patch centred on non-integer point
            x_poss = np.arange(minx - patch_offset, minx + patch_offset + 1, step=1, dtype=np.float32) + dx
            y_poss = np.arange(miny - patch_offset, miny + patch_offset + 1, step=1, dtype=np.float32) + dy
            interpolated_vals = np.zeros_like(x_poss)

            for j in range(len(x_poss)):
                x_floor = int(np.floor(x_poss[j]))
                y_floor = int(np.floor(y_poss[j]))
                x_ceil = int(np.ceil(x_poss[j]))
                y_ceil = int(np.ceil(y_poss[j]))
                top_left = image[x_floor, y_floor]
                top_right = image[x_ceil, y_floor]
                bottom_left = image[x_floor, y_ceil]
                bottom_right = image[x_ceil, y_ceil]
                interpolated_vals[j] = bilinear_interpolation(x_pos[i], y_pos[i], top_left, top_right, bottom_left,
                                                      bottom_right)

            means[i] = np.mean(interpolated_vals)


@nb.guvectorize([(nb.float64[:, :], nb.int16, nb.int16, nb.float32[:, :])], '(x,y),(),()->(x,y)', nopython=True)
def median_filter(image, kernel_size, padding, out_filtered):
    """
    Perform median filter on image and write to out_filtered
    :param image: Image to perform median filter on
    :param kernel_size: Kernel to use in median filter, usually 3
    :param padding: Padding size used on image. Note: Must be greater than (kernel_size - 1) / 2.
    :param out_filtered: Pass an initialised array with same dimensions as image. This becomes the median image.
    :return: No return, since we're using numba guvectorize, instead an initialised empty image must be passed into
            out_filtered and this value is updated by the function.
    """
    width, height = image.shape
    patch = int((kernel_size - 1) / 2)
    for x in range(padding, width - padding):
        for y in range(padding, height - padding):
            out_filtered[x, y] = np.median(image[x - patch:x + patch + 1, y - patch:y + patch + 1])


def bm3d_filter(image, sigma_psd=50/255):
    """
    Perform BM3D filter on image
    :param image: Image (numpy ndarray) to perform bm3d filter on
    :param sigma_psd: Standard deviation for intensities in range [0,255]
    :return: Filtered image
    """
    image_copy = image.copy()
    denoised_image = bm3d.bm3d(image_copy, sigma_psd=sigma_psd, stage_arg=bm3d.BM3DStages.ALL_STAGES)
    return denoised_image


@nb.njit('float64[:,:](uint8[:,:], float32, float32, float32)')
def homomorphic_filter(image, cutoff_freq=10, a=0.75, b=1.0):
    """
    Performs homomorphic filter on image using gaussian high-frequency filter
    Implementation is adapted from https://github.com/glasgio/homomorphic-filter
    :param image: uint8 ndarray to perform homomorphic filter on
    :param cutoff_freq: Cutoff frequency for Gaussian filter
    :param a, b: Floats to use on emphasis filter: H = a + b*H. Eg: a=0.5, b=1.5
    :return: Filtered image
    """

    # Replace values of 0 with the smallest >0 value, since log cannot be applied to 0.
    image.ravel()[image.ravel() == 0] = 1e-5
    # Take the image to log domain
    I_log = np.log1p(image)

    # Take image to frequency domain. Run in Object mode because numba does not support c-compiled np.fft functions.
    with nb.objmode(I_fft='complex128[:,:]'):
        I_fft = np.fft.fft2(I_log)

    # Apply Gaussian Mask
    P = I_fft.shape[0] / 2
    Q = I_fft.shape[1] / 2
    with nb.objmode(U='int32[:,:]', V='int32[:,:]'):
        U, V = np.meshgrid(range(I_fft.shape[0]), range(I_fft.shape[1]), sparse=False, indexing='ij')
    Duv = (U - P) ** 2 + (V - Q) ** 2
    H = np.exp((-Duv / (2 * cutoff_freq ** 2)))

    # Obtain high-pass filter by taking 1 and subtracting the low pass filter.
    H = 1 - H

    # Apply filter on frequency domain then take the image back to spatial domain
    with nb.objmode(H='float64[:,:]'):
        H = np.fft.fftshift(H)
    I_fft_filt = (a + b*H)*I_fft
    with nb.objmode(I_filt='complex128[:,:]'):
        I_filt = np.fft.ifft2(I_fft_filt)
    I = np.exp(np.real(I_filt)) - 1
    return I





