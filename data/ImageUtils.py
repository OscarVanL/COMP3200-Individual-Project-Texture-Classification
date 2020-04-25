import cv2
import numpy as np
import numba as nb
from skimage.util import random_noise


def convert_float32_image_uint8(image: np.ndarray):
    """
    Writes a float32 greyscale image_scaled normalised to zero mean and unit variance (in range [-1, 1]) to a [0, 255]
    range (useful for writing images with cv2.imwrite)
    Using code from Stack Overflow user: https://stackoverflow.com/a/50966901/6008271
    """
    norm_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return norm_image


def convert_uint8_image_float32(image: np.ndarray):
    """
    Converts a uint8 greyscale image_scaled (in range [0, 255]) to float32 in range [-1, 1] (unit variance).
    """
    norm_image = np.copy(image)
    norm_image = norm_image.astype(np.float32)
    norm_image = cv2.normalize(norm_image, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image.astype(np.float32)
    return norm_image


def scale_uint8_image_float32(image):
    """
    Converts and scales uint8 image_scaled to float32 image_scaled with zero mean and unit variance
    This gives it the range [-1, 1] and a mean of 0.
    """
    # Convert to float32
    scaled_img = np.copy(image)
    scaled_img = scaled_img.astype(np.float32)
    # Normalise image_scaled to zero-mean
    scaled_img -= image.mean()

    # Convert uint8 image_scaled to float32 in range [-1, 1] (unit variance)
    scaled_img = convert_uint8_image_float32(scaled_img)
    return scaled_img


@nb.njit('float32[:,:](float32[:,:], float32)')
def add_gaussian_noise_numba(image, sigma):
    """
    Credit: https://stackoverflow.com/a/30609854/6008271
    Uses 0-mean for gaussian noise.
    :param image: greyscale image_scaled as ndarray
    :param sigma: standard deviation of noise
    :return: noisy image_scaled
    """
    row, col = image.shape
    gauss = np.random.normal(0, sigma/255, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    return noisy.astype(np.float32)

def add_gaussian_noise_skimage(image, sigma):
    # Note: variance = (standard deviation) ** 2
    image_copy = image.copy()
    return random_noise(image_copy, mode='gaussian', clip=True, var=sigma/255)


@nb.njit('float32[:,:](float32[:,:], float32)')
def add_salt_pepper_noise_numba(image, amount):
    """
    Adapted from: https://stackoverflow.com/a/30609854/6008271
    Credit to @stuartarchibald for helping adapt this from using fancy indexing.
    Uses 50/50 salt-pepper split
    :param image: greyscale image_scaled as ndarray
    :param amount: ratio of pixels to give salt and pepper noise. eg: 0.02
    :return: noisy image_scaled
    """
    width, height = image.shape
    noisy = image.copy()
    num_noisy = np.ceil(amount * image.size * 0.5)

    n = int(num_noisy)
    salt_xs = np.random.randint(0, width - 1, n)
    salt_ys = np.random.randint(0, height - 1, n)
    pepper_xs = np.random.randint(0, width - 1, n)
    pepper_ys = np.random.randint(0, height - 1, n)

    for i in range(n):
        ix = salt_xs[i]
        iy = salt_ys[i]
        noisy[ix, iy] = 1
    for i in range(n):
        ix = pepper_xs[i]
        iy = pepper_ys[i]
        noisy[ix, iy] = -1

    return noisy

def add_salt_pepper_noise_skimage(image, amount):
    return random_noise(image, mode='s&p', amount=amount)


@nb.njit('float32[:,:](float32[:,:], float32)')
def add_speckle_noise_numba(image, var):
    """
    Add speckle noise to an image_scaled
    :param image: Image to add noise to
    :param var: Variance of noise (eg: 0.04)
    :return: noisy image_scaled
    """
    row, col = image.shape
    gauss = np.random.randn(row, col).astype(np.float32)
    gauss = gauss.reshape(row, col)
    return image + (image * gauss * var)

def add_speckle_noise_skimage(image, var):
    # For some reason, speckle noise does not apply properly in range [-1, 1]...
    image_copy = image.copy()
    norm_image = cv2.normalize(image_copy, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    noisy_image = random_noise(norm_image, mode='speckle', clip=True, var=var)
    return cv2.normalize(noisy_image, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
