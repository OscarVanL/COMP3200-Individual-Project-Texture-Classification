import timeit

# Test 1: Is old median function, new median or new median+parallel faster?
def test1():
    setup1 = """
import os
import cv2
import numpy as np
import numba as nb
import algorithms.MRELBP as MRELBP
KYLBERG_BLANKET_DIR = os.path.join(os.getcwd(), 'data', 'kylberg', 'blanket1', 'blanket1-a-p001.png')

image_scaled = cv2.imread(KYLBERG_BLANKET_DIR, cv2.IMREAD_GRAYSCALE)
image_scaled = MRELBP.MedianRobustExtendedLBP.scale_img(image_scaled)

padding = 1


def median_filter(image_scaled: np.ndarray, kernel_size):
    patch = int((kernel_size - 1) / 2)
    width, height = image_scaled.shape

    if kernel_size % 2 == 0:
        raise ValueError('median_filter() kernel_size should be an odd number.')
    if kernel_size > width - padding or kernel_size > height - padding:
        raise ValueError('Kernel size for Median Filter is larger than Image')

    # Allocate memory for output image_scaled
    filtered_image = np.zeros_like(image_scaled, dtype=np.float32)

    # Todo: Can this be completed with Broadcasting for speed boost?
    median(image_scaled, padding, patch, width, height, filtered_image)

    return filtered_image

def median_filter_parallel(image_scaled: np.ndarray, kernel_size):
    patch = int((kernel_size - 1) / 2)
    width, height = image_scaled.shape

    if kernel_size % 2 == 0:
        raise ValueError('median_filter() kernel_size should be an odd number.')
    if kernel_size > width - padding or kernel_size > height - padding:
        raise ValueError('Kernel size for Median Filter is larger than Image')

    # Allocate memory for output image_scaled
    filtered_image = np.zeros_like(image_scaled, dtype=np.float32)

    # Todo: Can this be completed with Broadcasting for speed boost?
    median_parallel(image_scaled, padding, patch, width, height, filtered_image)

    return filtered_image


def median_filter_old(image_scaled: np.ndarray, kernel_size):
    patch = int((kernel_size - 1) / 2)
    width, height = image_scaled.shape

    if kernel_size % 2 == 0:
        raise ValueError('median_filter() kernel_size should be an odd number.')
    if kernel_size > width - padding or kernel_size > height - padding:
        raise ValueError('Kernel size for Median Filter is larger than Image')

    # Allocate memory for output image_scaled
    filtered_image = np.zeros_like(image_scaled, dtype=np.float32)

    # Todo: Can this be completed with Broadcasting for speed boost?
    for x in range(padding, width - padding - 1):
        for y in range(padding, height - padding - 1):
            # Iterate through each pixel in the image_scaled. Take the kernel and calculate the median value.
            kernel = image_scaled[x - patch:x + patch + 1, y - patch:y + patch + 1]
            median = np.median(kernel.flatten())

            filtered_image[x][y] = median

    return filtered_image


@nb.guvectorize([(nb.float64[:, :], nb.int32, nb.int32, nb.int32, nb.int32, nb.float32[:, :])],
                '(x,y),(),(),(),()->(x,y)', nopython=True)
def median(image_scaled, padding, patch, width, height, median):
    for x in range(padding, width - padding - 1):
        for y in range(padding, height - padding - 1):
            median[x, y] = np.median(image_scaled[x - patch:x + patch + 1, y - patch:y + patch + 1])

@nb.guvectorize([(nb.float64[:, :], nb.int32, nb.int32, nb.int32, nb.int32, nb.float32[:, :])],
                '(x,y),(),(),(),()->(x,y)', nopython=True, target='parallel')
def median_parallel(image_scaled, padding, patch, width, height, median):
    for x in range(padding, width - padding - 1):
        for y in range(padding, height - padding - 1):
            median[x, y] = np.median(image_scaled[x - patch:x + patch + 1, y - patch:y + patch + 1])
    """

    print("Old")
    print(timeit.timeit("median_filter_old(image_scaled, 3)", setup=setup1, number=5))

    print("New")
    print(timeit.timeit("median_filter(image_scaled, 3)", setup=setup1, number=5))

    print("New parallel")
    print(timeit.timeit("median_filter_parallel(image_scaled, 3)", setup=setup1, number=5))

# Test 2: Is CPU, parallel or GPU numba bilinear faster?
def test2():
    setup2 = """
import algorithms.SharedFunctions as SharedFunctions
import numpy as np
import math

x_pos = 5.4
y_pos = 5.5
minx, miny = math.floor(x_pos), math.floor(y_pos)
dx, dy = x_pos - minx, y_pos - miny
patch_offset = 3

x_poss = np.arange(minx - patch_offset, minx + patch_offset + 1, step=1, dtype=np.float32) + dx
y_poss = np.arange(miny - patch_offset, miny + patch_offset + 1, step=1, dtype=np.float32) + dy
top_left = np.arange(0, 1, 0.05, dtype=np.float32)
top_right = np.arange(0, 1, 0.05, dtype=np.float32)
bottom_left = np.arange(0, 1, 0.05, dtype=np.float32)
bottom_right = np.arange(0, 1, 0.05, dtype=np.float32)
    """

    print("CPU")
    print(timeit.timeit(
        "SharedFunctions.bilinear_interpolation(x_pos, y_pos, top_left, top_right, bottom_left, bottom_right)",
        setup=setup2, number=1000))

    print("Parallel")
    print(timeit.timeit(
        "SharedFunctions.bilinear_interpolation_parallel(x_pos, y_pos, top_left, top_right, bottom_left, bottom_right)",
        setup=setup2, number=1000))

    print("CUDA")
    print(timeit.timeit(
        "SharedFunctions.bilinear_interpolation_gpu(x_pos, y_pos, top_left, top_right, bottom_left, bottom_right)",
        setup=setup2, number=1000))

# Test 3: Is old or new radial coordinate calculation faster?
def test3():
    setup3 = """
import numpy as np
x_c = 50
y_c = 50
radial_angles = [45, 90, 135, 180, 225, 270, 315, 0]
r1 = 3
r2 = 5

def old_radial_calc():
    r1_coords = list(
        zip(x_c + r1 * np.cos(radial_angles), y_c + r1 * np.sin(radial_angles)))
    r2_coords = list(zip(x_c + r2 * np.cos(radial_angles), y_c + r2 * np.sin(radial_angles)))
    """

    setup4 = """
import numpy as np
from algorithms.MRELBP import MedianRobustExtendedLBP
import math
x_c = 50
y_c = 50
radial_angles = (np.arange(0, 8) * -(2 * math.pi) / 8).astype(np.float32)
r1 = 3
r2 = 5
x1s = np.zeros_like(radial_angles)
y1s = np.zeros_like(radial_angles)
x2s = np.zeros_like(radial_angles)
y2s = np.zeros_like(radial_angles)
    """

    print("Old")
    print(timeit.timeit(
        "old_radial_calc()",
        setup=setup3, number=1000))

    print("New")
    print(timeit.timeit(
        "MedianRobustExtendedLBP.calculate_radial_coords(x_c, y_c, r1, r2, radial_angles, x1s, y1s, x2s, y2s)",
        setup=setup4, number=1000))

#test3()




def test_noise():
    setup5 = """
import os
import cv2
import numpy as np
import numba as nb
import data.ImageUtils as ImageUtils

KYLBERG_BLANKET_DIR = os.path.join(os.getcwd(), 'data', 'kylberg', 'blanket1', 'blanket1-a-p001.png')

image_scaled = cv2.imread(KYLBERG_BLANKET_DIR, cv2.IMREAD_GRAYSCALE)
image_scaled = ImageUtils.scale_img(image_scaled)
    """
    print("Gaussian")
    print(timeit.timeit("ImageUtils.add_gaussian_noise(image_scaled, 10)", setup=setup5, number=100))

    print("Speckle")
    print(timeit.timeit("ImageUtils.add_speckle_noise(image_scaled)", setup=setup5, number=100))

    print("S&P")
    print(timeit.timeit("ImageUtils.add_salt_pepper_noise(image_scaled, 0.02)", setup=setup5, number=100))

    print("S&P Numba")
    print(timeit.timeit("ImageUtils.add_salt_pepper_noise_numba(image_scaled, 0.02)", setup=setup5, number=100))


#test_noise()


import os
import cv2
from algorithms import BM3DELBP, SARBM3D
import data.ImageUtils as ImageUtils
from data.DatasetManager import Image
from example import GenerateExamples
import numpy as np

KYLBERG_BLANKET_DIR = os.path.join(os.getcwd(), 'data', 'kylberg', 'blanket1', 'blanket1-a-p001.png')

algorithm = BM3DELBP.NoiseClassifier()

image_uint8 = cv2.resize(cv2.imread(KYLBERG_BLANKET_DIR, cv2.IMREAD_GRAYSCALE), (0, 0),
                                fx=0.5,
                                fy=0.5)
image_scaled = ImageUtils.scale_uint8_image_float32(image_uint8)

#image_gaussian_25 = ImageUtils.add_gaussian_noise(image_scaled, 25)
image_speckle_002 = ImageUtils.add_speckle_noise(image_scaled, 0.02)
#image_salt_pepper_002 = ImageUtils.add_salt_pepper_noise(image_scaled, 0.02)

# print("NORMAL IMAGE, NO ADDED NOISE")
# image_nonoise = Image(image_scaled, 'blanket1-a-p001.png', 'blanket1')
# algorithm.describe(image_nonoise, test_image=False)
# print("GAUSSIAN SIGMA 25")
# image_gaussian = Image(image_gaussian_25, 'blanket1-a-p001.png', 'blanket1')
# algorithm.describe(image_gaussian, test_image=False)
# print("SPECKLE NOISE")
# image_speckle = Image(image_speckle_002, 'blanket1-a-p001.png', 'blanket1')
# algorithm.describe(image_speckle, test_image=False)
# print("SALT PEPPER NOISE")
# image_salt_pepper = Image(image_salt_pepper_002, 'blanket1-a-p001.png', 'blanket1')
# algorithm.describe(image_salt_pepper, test_image=False)


image_speckle_002 = ImageUtils.add_speckle_noise(image_uint8.astype(np.float32), 0.02)
print("SARBM3D FILTERING ON SPECKLE IMAGE")
L=50
sarbm3d = SARBM3D.SARBM3DFilter()
sarbm3d_filtered = sarbm3d.sar_bm3d_filter(image_speckle_002, 'blanket1-a-p001', L)
sarbm3d.disconnect_matlab()

noise_estimate = image_speckle_002 - sarbm3d_filtered

GenerateExamples.write_image(ImageUtils.convert_float32_image_uint8(image_scaled), 'sarbm3d', 'original.png')
GenerateExamples.write_image(ImageUtils.convert_float32_image_uint8(image_speckle_002), 'sarbm3d', 'before (with speckle noise).png')
GenerateExamples.write_image(ImageUtils.convert_float32_image_uint8(sarbm3d_filtered), 'sarbm3d', 'sarbm3d-filtered-L_{}.png'.format(L))
GenerateExamples.write_image(ImageUtils.convert_float32_image_uint8(noise_estimate), 'sarbm3d', 'noise estimate-L_{}.png'.format(L))