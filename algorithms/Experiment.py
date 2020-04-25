import timeit

# Test 1: Is old median function, new median or new median+parallel faster?
def test1():
    setup1 = """
import os
import cv2
import numpy as np
import numba as nb
import algorithms.MRELBP as MRELBP
KYLBERG_BLANKET_001 = os.path.join(os.getcwd(), 'data', 'kylberg', 'blanket1', 'blanket1-a-p001.png')

image_scaled = cv2.imread(KYLBERG_BLANKET_001, cv2.IMREAD_GRAYSCALE)
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

KYLBERG_BLANKET_001 = os.path.join(os.getcwd(), 'data', 'kylberg', 'blanket1', 'blanket1-a-p001.png')

image_scaled = cv2.imread(KYLBERG_BLANKET_001, cv2.IMREAD_GRAYSCALE)
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

#
# def compare_images(img1, img2):
#     # calculate the difference and its norms
#     diff = img1 - img2  # elementwise for scipy arrays
#     m_norm = sum(abs(diff))  # Manhattan norm
#     z_norm = norm(diff.ravel(), 0)  # Zero norm
#     return (m_norm, z_norm)
#
# #algorithm = BM3DELBP.NoiseClassifier()
#
# image_uint8 = cv2.resize(cv2.imread(KYLBERG_BLANKET_001, cv2.IMREAD_GRAYSCALE), (0, 0),
#                                 fx=1.0,
#                                 fy=1.0)
# print(image_uint8.dtype)
# image_scaled = ImageUtils.scale_uint8_image_float32(image_uint8)
# image_speckle_uint8 = ImageUtils.add_speckle_noise_skimage(image_uint8, 0.02)
# image_gaussian_10 = ImageUtils.add_gaussian_noise_skimage(image_scaled, 10)
# image_salt_pepper = ImageUtils.add_salt_pepper_noise_skimage(image_scaled, 0.02)
#
# test_image = Image(image_salt_pepper, None, None)
# bm3d_test_image = BM3DELBP.BM3DELBPImage(test_image)
#
# classifier = NoiseClassifier.NoiseClassifier()
#
# GlobalConfig.set('debug', True)
#
# classifier.describe(test_image, test_image=False)


import os
import cv2
from algorithms import BM3DELBP, SARBM3D
import data.ImageUtils as ImageUtils
from data.DatasetManager import Image
from example import GenerateExamples
import numpy as np
from algorithms import SharedFunctions, NoiseClassifier
from scipy.linalg import norm
from scipy import sum
from config import GlobalConfig
import math

prefix = os.path.join(os.getcwd(), 'data', 'kylberg')






from scipy.stats import skew, kurtosis

def tune_homomorphic_filter():
    textures = []

    textures.append(os.path.join(prefix, 'blanket1', 'blanket1-a-p001.png'))
    textures.append(os.path.join(prefix, 'blanket2', 'blanket2-a-p001.png'))
    textures.append(os.path.join(prefix, 'canvas1', 'canvas1-a-p001.png'))
    textures.append(os.path.join(prefix, 'ceiling1', 'ceiling1-a-p001.png'))
    textures.append(os.path.join(prefix, 'ceiling2', 'ceiling2-a-p001.png'))
    textures.append(os.path.join(prefix, 'cushion1', 'cushion1-a-p001.png'))
    textures.append(os.path.join(prefix, 'floor1', 'floor1-a-p001.png'))
    textures.append(os.path.join(prefix, 'floor2', 'floor2-a-p001.png'))
    textures.append(os.path.join(prefix, 'grass1', 'grass1-a-p001.png'))
    textures.append(os.path.join(prefix, 'lentils1', 'lentils1-a-p001.png'))
    textures.append(os.path.join(prefix, 'linseeds1', 'linseeds1-a-p001.png'))
    textures.append(os.path.join(prefix, 'oatmeal1', 'oatmeal1-a-p001.png'))
    textures.append(os.path.join(prefix, 'pearlsugar1', 'pearlsugar1-a-p001.png'))
    textures.append(os.path.join(prefix, 'rice1', 'rice1.b-p001.png'))
    textures.append(os.path.join(prefix, 'rice2', 'rice2-a-p001.png'))
    textures.append(os.path.join(prefix, 'rug1', 'rug1-a-p001.png'))
    textures.append(os.path.join(prefix, 'sand1', 'sand1-a-p001.png'))
    textures.append(os.path.join(prefix, 'scarf1', 'scarf1-a-p001.png'))
    textures.append(os.path.join(prefix, 'scarf2', 'scarf2-a-p001.png'))
    textures.append(os.path.join(prefix, 'screen1', 'screen1-a-p001.png'))
    textures.append(os.path.join(prefix, 'seat1', 'seat1-a-p001.png'))
    textures.append(os.path.join(prefix, 'seat2', 'seat2-a-p001.png'))
    textures.append(os.path.join(prefix, 'sesameseeds1', 'sesameseeds1-a-p001.png'))
    textures.append(os.path.join(prefix, 'stone1', 'stone1-a-p001.png'))
    textures.append(os.path.join(prefix, 'stone2', 'stone2-a-p001.png'))
    textures.append(os.path.join(prefix, 'stone3', 'stone3-a-p001.png'))
    textures.append(os.path.join(prefix, 'stoneslab1', 'stoneslab1-a-p001.png'))
    textures.append(os.path.join(prefix, 'wall1', 'wall1-a-p001.png'))

    cutoff_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 5, 10]
    a_vals = [0.6, 0.75, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    b_vals = [0.1, 0.2, 0.4, 0.6, 0.75, 1.0, 1.25, 1.5]

    diffs = [0] * len(cutoff_vals) * len(a_vals) * len(b_vals)
    cutoff_vals_order = [0] * len(cutoff_vals) * len(a_vals) * len(b_vals)
    a_vals_order = [0] * len(cutoff_vals) * len(a_vals) * len(b_vals)
    b_vals_order = [0] * len(cutoff_vals) * len(a_vals) * len(b_vals)

    for texture in textures:

        image_uint8 = cv2.resize(cv2.imread(texture, cv2.IMREAD_GRAYSCALE), (0, 0),
                                 fx=0.5,
                                 fy=0.5)
        image_scaled = ImageUtils.scale_uint8_image_float32(image_uint8)
        # image_salt_pepper = ImageUtils.add_salt_pepper_noise_skimage(image_scaled, 0.02)
        # image_salt_pepper = ImageUtils.convert_float32_image_uint8(image_salt_pepper)
        image_gaussian = ImageUtils.add_gaussian_noise_skimage(image_scaled, 10)
        image_gaussian = ImageUtils.convert_float32_image_uint8(image_gaussian)
        image_speckle = ImageUtils.add_speckle_noise_skimage(image_scaled, 0.02)
        image_speckle = ImageUtils.convert_float32_image_uint8(image_speckle)

        diff_index = 0

        for cut in cutoff_vals:
            for a in a_vals:
                for b in b_vals:

                    # homomorphic_salt_pepper = SharedFunctions.homomorphic_filter(image_salt_pepper, cut, a, b)
                    # homomorphic_salt_pepper = ImageUtils.convert_uint8_image_float32(homomorphic_salt_pepper)
                    homomorphic_gaussian = SharedFunctions.homomorphic_filter(image_gaussian, cut, a, b)
                    homomorphic_gaussian = ImageUtils.convert_uint8_image_float32(homomorphic_gaussian)
                    homomorphic_speckle = SharedFunctions.homomorphic_filter(image_speckle, cut, a, b)
                    homomorphic_speckle = ImageUtils.convert_uint8_image_float32(homomorphic_speckle)

                    # noise_salt_pepper = image_scaled - homomorphic_salt_pepper
                    noise_gaussian = image_scaled - homomorphic_gaussian
                    noise_speckle = image_scaled - homomorphic_speckle

                    # kurtosis_sp = kurtosis(a=noise_salt_pepper, axis=None, fisher=False, nan_policy='raise')
                    # skewness_sp = skew(a=noise_salt_pepper, axis=None, nan_policy='raise')

                    kurtosis_gauss = kurtosis(a=noise_gaussian, axis=None, fisher=False, nan_policy='raise')
                    skewness_gauss = skew(a=noise_gaussian, axis=None, nan_policy='raise')

                    kurtosis_speckle = kurtosis(a=noise_speckle, axis=None, fisher=False, nan_policy='raise')
                    skewness_speckle = skew(a=noise_speckle, axis=None, nan_policy='raise')

                    diff = abs(kurtosis_speckle - kurtosis_gauss) + abs(skewness_speckle - skewness_gauss) # + math.sqrt(abs(kurtosis_speckle - kurtosis_sp))
                    print("cutoff: {}, a: {}, b: {}, diff: {}".format(cut, a, b, diff))

                    # Increment the difference value for that configuration index
                    diffs[diff_index] += diff
                    cutoff_vals_order[diff_index] = cut
                    a_vals_order[diff_index] = a
                    b_vals_order[diff_index] = b

                    diff_index += 1

    index_best = np.argmax(diffs)
    print("best index: {}".format(index_best))
    best_diff = diffs[index_best]
    best_cutoff = cutoff_vals_order[index_best]
    best_a = a_vals_order[index_best]
    best_b = b_vals_order[index_best]
    print("OVERALL BEST: Cutoff: {}, a: {}, b: {}, difference: {}".format(best_cutoff, best_a, best_b, best_diff))

    image_uint8 = cv2.resize(cv2.imread(textures[0], cv2.IMREAD_GRAYSCALE), (0, 0),
                             fx=0.5,
                             fy=0.5)
    image_scaled = ImageUtils.scale_uint8_image_float32(image_uint8)
    image_salt_pepper = ImageUtils.add_salt_pepper_noise_skimage(image_scaled, 0.02)
    image_salt_pepper = ImageUtils.convert_float32_image_uint8(image_salt_pepper)
    image_gaussian = ImageUtils.add_gaussian_noise_skimage(image_scaled, 10)
    image_gaussian = ImageUtils.convert_float32_image_uint8(image_gaussian)
    image_speckle = ImageUtils.add_speckle_noise_skimage(image_uint8, 0.02)
    image_speckle = ImageUtils.convert_float32_image_uint8(image_speckle)

    homomorphic_salt_pepper = SharedFunctions.homomorphic_filter(image_salt_pepper, best_cutoff, best_a, best_b)
    homomorphic_gaussian = SharedFunctions.homomorphic_filter(image_gaussian, best_cutoff, best_a, best_b)
    homomorphic_speckle = SharedFunctions.homomorphic_filter(image_speckle, best_cutoff, best_a, best_b)
    GenerateExamples.write_image(ImageUtils.convert_float32_image_uint8(homomorphic_salt_pepper), 'Homomorphic', 'homomorphic-salt-pepper.png')
    GenerateExamples.write_image(ImageUtils.convert_float32_image_uint8(homomorphic_gaussian), 'Homomorphic', 'homomorphic-gaussian.png')
    GenerateExamples.write_image(ImageUtils.convert_float32_image_uint8(homomorphic_speckle), 'Homomorphic', 'homomorphic-speckle.png')

    homomorphic_salt_pepper = ImageUtils.convert_uint8_image_float32(homomorphic_salt_pepper)
    homomorphic_gaussian = ImageUtils.convert_uint8_image_float32(homomorphic_gaussian)
    homomorphic_speckle = ImageUtils.convert_uint8_image_float32(homomorphic_speckle)

    noise_salt_pepper = image_scaled - homomorphic_salt_pepper
    noise_gaussian = image_scaled - homomorphic_gaussian
    noise_speckle = image_scaled - homomorphic_speckle

    GenerateExamples.write_image(ImageUtils.convert_float32_image_uint8(noise_salt_pepper), 'Homomorphic', 'noise_estimate_salt_pepper.png')
    GenerateExamples.write_image(ImageUtils.convert_float32_image_uint8(noise_gaussian), 'Homomorphic', 'noise_estimate_gaussian.png')
    GenerateExamples.write_image(ImageUtils.convert_float32_image_uint8(noise_speckle), 'Homomorphic', 'noise_estimate_speckle.png')

    kurtosis_sp = kurtosis(a=noise_salt_pepper, axis=None, fisher=False, nan_policy='raise')
    skewness_sp = skew(a=noise_salt_pepper, axis=None, nan_policy='raise')

    kurtosis_gauss = kurtosis(a=noise_gaussian, axis=None, fisher=False, nan_policy='raise')
    skewness_gauss = skew(a=noise_gaussian, axis=None, nan_policy='raise')

    kurtosis_speckle = kurtosis(a=noise_speckle, axis=None, fisher=False, nan_policy='raise')
    skewness_speckle = skew(a=noise_speckle, axis=None, nan_policy='raise')

    print("Salt & Pepper: Skewness {}, Kurtosis {}".format(skewness_sp, kurtosis_sp))
    print("Gaussian: Skewness {}, Kurtosis {}".format(skewness_gauss, kurtosis_gauss))
    print("Speckle: Skewness {}, Kurtosis {}".format(skewness_speckle, kurtosis_speckle))


#tune_homomorphic_filter()


def tune_bm3d_filter():
    textures = []

    textures.append(os.path.join(prefix, 'blanket1', 'blanket1-a-p001.png'))
    textures.append(os.path.join(prefix, 'blanket2', 'blanket2-a-p001.png'))
    textures.append(os.path.join(prefix, 'canvas1', 'canvas1-a-p001.png'))
    textures.append(os.path.join(prefix, 'ceiling1', 'ceiling1-a-p001.png'))
    textures.append(os.path.join(prefix, 'ceiling2', 'ceiling2-a-p001.png'))
    textures.append(os.path.join(prefix, 'cushion1', 'cushion1-a-p001.png'))
    textures.append(os.path.join(prefix, 'floor1', 'floor1-a-p001.png'))
    textures.append(os.path.join(prefix, 'floor2', 'floor2-a-p001.png'))
    textures.append(os.path.join(prefix, 'grass1', 'grass1-a-p001.png'))
    textures.append(os.path.join(prefix, 'lentils1', 'lentils1-a-p001.png'))
    textures.append(os.path.join(prefix, 'linseeds1', 'linseeds1-a-p001.png'))
    textures.append(os.path.join(prefix, 'oatmeal1', 'oatmeal1-a-p001.png'))
    textures.append(os.path.join(prefix, 'pearlsugar1', 'pearlsugar1-a-p001.png'))
    textures.append(os.path.join(prefix, 'rice1', 'rice1.b-p001.png'))
    textures.append(os.path.join(prefix, 'rice2', 'rice2-a-p001.png'))
    textures.append(os.path.join(prefix, 'rug1', 'rug1-a-p001.png'))
    textures.append(os.path.join(prefix, 'sand1', 'sand1-a-p001.png'))
    textures.append(os.path.join(prefix, 'scarf1', 'scarf1-a-p001.png'))
    textures.append(os.path.join(prefix, 'scarf2', 'scarf2-a-p001.png'))
    textures.append(os.path.join(prefix, 'screen1', 'screen1-a-p001.png'))
    textures.append(os.path.join(prefix, 'seat1', 'seat1-a-p001.png'))
    textures.append(os.path.join(prefix, 'seat2', 'seat2-a-p001.png'))
    textures.append(os.path.join(prefix, 'sesameseeds1', 'sesameseeds1-a-p001.png'))
    textures.append(os.path.join(prefix, 'stone1', 'stone1-a-p001.png'))
    textures.append(os.path.join(prefix, 'stone2', 'stone2-a-p001.png'))
    textures.append(os.path.join(prefix, 'stone3', 'stone3-a-p001.png'))
    textures.append(os.path.join(prefix, 'stoneslab1', 'stoneslab1-a-p001.png'))
    textures.append(os.path.join(prefix, 'wall1', 'wall1-a-p001.png'))

    sigma = [70, 120]

    diffs = [0] * len(sigma)

    for texture in textures:

        image_uint8 = cv2.resize(cv2.imread(texture, cv2.IMREAD_GRAYSCALE), (0, 0),
                                 fx=0.5,
                                 fy=0.5)
        image_scaled = ImageUtils.scale_uint8_image_float32(image_uint8)
        image_gaussian = ImageUtils.add_gaussian_noise_skimage(image_scaled, 10)
        image_gaussian = ImageUtils.convert_float32_image_uint8(image_gaussian)
        image_speckle = ImageUtils.add_speckle_noise_skimage(image_scaled, 0.02)
        image_speckle = ImageUtils.convert_float32_image_uint8(image_speckle)

        diff_index = 0

        for val in sigma:
            sigma_val = val/255
            print("Evaluating {}/255".format(val))
            bm3d_gaussian = SharedFunctions.bm3d_filter(image_gaussian, sigma_psd=sigma_val)
            bm3d_gaussian = ImageUtils.convert_uint8_image_float32(bm3d_gaussian)
            bm3d_speckle = SharedFunctions.bm3d_filter(image_speckle, sigma_psd=sigma_val)
            bm3d_speckle = ImageUtils.convert_uint8_image_float32(bm3d_speckle)

            # noise_salt_pepper = image_scaled - homomorphic_salt_pepper
            noise_gaussian = image_scaled - bm3d_gaussian
            noise_speckle = image_scaled - bm3d_speckle

            kurtosis_gauss = kurtosis(a=noise_gaussian, axis=None, fisher=False, nan_policy='raise')
            skewness_gauss = skew(a=noise_gaussian, axis=None, nan_policy='raise')
            kurtosis_speckle = kurtosis(a=noise_speckle, axis=None, fisher=False, nan_policy='raise')
            skewness_speckle = skew(a=noise_speckle, axis=None, nan_policy='raise')

            diff = abs(kurtosis_speckle - kurtosis_gauss) + abs(skewness_speckle - skewness_gauss) # + math.sqrt(abs(kurtosis_speckle - kurtosis_sp))
            print("sigma_psd: {}/255, diff: {}".format(val, diff))

            # Increment the difference value for that configuration index
            diffs[diff_index] += diff

            diff_index += 1

    index_best = np.argmax(diffs)
    print("best index: {}".format(index_best))
    best_diff = diffs[index_best]
    best_sigma = sigma[index_best]
    print("OVERALL BEST: sigma_val: {}/255, difference: {}".format(best_sigma, best_diff))

    image_uint8 = cv2.resize(cv2.imread(textures[0], cv2.IMREAD_GRAYSCALE), (0, 0),
                             fx=0.5,
                             fy=0.5)
    image_scaled = ImageUtils.scale_uint8_image_float32(image_uint8)
    image_salt_pepper = ImageUtils.add_salt_pepper_noise_skimage(image_scaled, 0.02)
    image_salt_pepper = ImageUtils.convert_float32_image_uint8(image_salt_pepper)
    image_gaussian = ImageUtils.add_gaussian_noise_skimage(image_scaled, 10)
    image_gaussian = ImageUtils.convert_float32_image_uint8(image_gaussian)
    image_speckle = ImageUtils.add_speckle_noise_skimage(image_uint8, 0.02)
    image_speckle = ImageUtils.convert_float32_image_uint8(image_speckle)

    bm3d_salt_pepper = SharedFunctions.bm3d_filter(image_salt_pepper, sigma_psd=(best_sigma/255))
    bm3d_gaussian = SharedFunctions.bm3d_filter(image_gaussian, sigma_psd=(best_sigma/255))
    bm3d_speckle = SharedFunctions.bm3d_filter(image_speckle, sigma_psd=(best_sigma/255))
    GenerateExamples.write_image(ImageUtils.convert_float32_image_uint8(bm3d_salt_pepper), 'bm3d', 'homomorphic-salt-pepper.png')
    GenerateExamples.write_image(ImageUtils.convert_float32_image_uint8(bm3d_gaussian), 'bm3d', 'homomorphic-gaussian.png')
    GenerateExamples.write_image(ImageUtils.convert_float32_image_uint8(bm3d_speckle), 'bm3d', 'homomorphic-speckle.png')

    bm3d_salt_pepper = ImageUtils.convert_uint8_image_float32(bm3d_salt_pepper)
    bm3d_gaussian = ImageUtils.convert_uint8_image_float32(bm3d_gaussian)
    bm3d_speckle = ImageUtils.convert_uint8_image_float32(bm3d_speckle)

    noise_salt_pepper = image_scaled - bm3d_salt_pepper
    noise_gaussian = image_scaled - bm3d_gaussian
    noise_speckle = image_scaled - bm3d_speckle

    GenerateExamples.write_image(ImageUtils.convert_float32_image_uint8(noise_salt_pepper), 'bm3d', 'noise_estimate_salt_pepper.png')
    GenerateExamples.write_image(ImageUtils.convert_float32_image_uint8(noise_gaussian), 'bm3d', 'noise_estimate_gaussian.png')
    GenerateExamples.write_image(ImageUtils.convert_float32_image_uint8(noise_speckle), 'bm3d', 'noise_estimate_speckle.png')

    kurtosis_sp = kurtosis(a=noise_salt_pepper, axis=None, fisher=False, nan_policy='raise')
    skewness_sp = skew(a=noise_salt_pepper, axis=None, nan_policy='raise')

    kurtosis_gauss = kurtosis(a=noise_gaussian, axis=None, fisher=False, nan_policy='raise')
    skewness_gauss = skew(a=noise_gaussian, axis=None, nan_policy='raise')

    kurtosis_speckle = kurtosis(a=noise_speckle, axis=None, fisher=False, nan_policy='raise')
    skewness_speckle = skew(a=noise_speckle, axis=None, nan_policy='raise')

    print("Salt & Pepper: Skewness {}, Kurtosis {}".format(skewness_sp, kurtosis_sp))
    print("Gaussian: Skewness {}, Kurtosis {}".format(skewness_gauss, kurtosis_gauss))
    print("Speckle: Skewness {}, Kurtosis {}".format(skewness_speckle, kurtosis_speckle))


#tune_bm3d_filter()

from data import DatasetManager
from other import istarmap
import tqdm
from multiprocessing import Pool
from itertools import repeat

def generate_noise(noise_classifier, image : BM3DELBP.BM3DELBPImage):
    image.generate_gauss_10(noise_classifier)
    image.generate_speckle(noise_classifier)
    image.generate_salt_pepper_002(noise_classifier)


def tune_noise_classifier():
    GlobalConfig.set('dataset', 'kylberg')
    #GlobalConfig.set('ECS', True)
    GlobalConfig.set('algorithm', 'NoiseClassifier')
    GlobalConfig.set('scale', 0.5)
    GlobalConfig.set('CWD', os.getcwd())
    GlobalConfig.set('folds', 10)

    dataset = DatasetManager.KylbergTextures(num_classes=28, data_ratio=0.15)
    images = dataset.load_data()
    bm3d_images = []
    # Convert to BM3D images
    for image in images:
        new_image = BM3DELBP.BM3DELBPImage(image)
        bm3d_images.append(new_image)

    print("Image dataset loaded, loaded {} images", len(images))

    noise_classifier = NoiseClassifier.NoiseClassifier()

    with Pool(processes=GlobalConfig.get('cpu_count')) as pool:
        # Generate image featurevectors and replace DatasetManager.Image with BM3DELBP.BM3DELBPImage
        processed_dataset = []

        for image in tqdm.tqdm(
                pool.istarmap(generate_noise, zip(repeat(noise_classifier), images)),
                total=len(images), desc='Applying noise'):
            processed_dataset.append(image)
        images = processed_dataset

    print("Image noise generated")

    bm3d_sigma = [2, 5, 10, 15, 20, 30, 35, 40, 50]
    homomorphic_cutoff = [0.1, 0.5, 0.75, 1, 5, 10, 20]
    homomorphic_a = [0.5, 0.75, 1.0, 1.2]
    homomorphic_b = [0.1, 0.3, 0.5, 0.8, 1.0, 1.25]

    for sigma_val in bm3d_sigma:
        sigma = sigma_val / 255
        for cutoff in homomorphic_cutoff:
            for a in homomorphic_a:
                for b in homomorphic_b:
                    pass

    results = []  # List of tuples (F1, sigma_val, cutoff, a, b)

    # Sort largest to smallest, by F1 score
    results.sort(key=lambda tup: tup[0], reverse=True)

    print("Finished tuning parameters.")
    print("The top 3 results were:")
    f1, sigma, cutoff, a, b = results[0]
    print("F1: {}, sigma_val: {}, cutoff_freq: {}, a: {}, b: {}".format(f1, sigma, cutoff, a, b))
    f1, sigma, cutoff, a, b = results[1]
    print("F1: {}, sigma_val: {}, cutoff_freq: {}, a: {}, b: {}".format(f1, sigma, cutoff, a, b))
    f1, sigma, cutoff, a, b = results[2]
    print("F1: {}, sigma_val: {}, cutoff_freq: {}, a: {}, b: {}".format(f1, sigma, cutoff, a, b))



if __name__ == '__main__':
    tune_noise_classifier()