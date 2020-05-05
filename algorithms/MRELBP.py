from typing import List, Tuple

import numpy as np
from skimage.util import pad
import math
import numba as nb
from itertools import zip_longest
from sklearn.svm import SVC

from algorithms.AlgorithmInterfaces import ImageProcessorInterface, ImageClassifierInterface
from data import DatasetManager, ImageUtils
from algorithms import SharedFunctions
from example import GenerateExamples
from config import GlobalConfig


class MedianRobustExtendedLBP(ImageProcessorInterface):
    """
    Implementation of paper Median Robust Extended Local Binary Pattern for Texture Classification
    https://ieeexplore.ieee.org/document/7393828

    Useful references for similar implementations (in other languages):
    riu2 mapping encoding:
    https://github.com/javimazzaf/QuRVA/blob/master/getmapping.m
    LBP and MRELBP implementation in c++:
    https://git.io/Jv2P2
    """

    def __init__(self, r1=None, p=8, w_center=3, w_r1=None, save_img=False):
        """
        :param r1: int or [int]. Radius(s) to use in RELBP_NI and RELBP_RD descriptors
        :param p: int. Number of neighbours for LBP
        :param w_center: int. Patch size for median filter and calculating RELBP_CI. Must be odd.
        :param w_r1: int or [int]. Patch size to use for each value of r1. Length must match r1's. Must be odd.
        """
        # If not assigned, initialise to same values as in paper.
        super().__init__(save_img)
        if r1 is None:
            r1 = [2, 4, 6, 8]
        if w_r1 is None:
            w_r1 = [3, 5, 7, 9]
        # 'r' used in MRELBP function as larger kernel r (sometimes called r_1 in paper)
        if isinstance(r1, int):
            r1 = [r1]
        self.r1 = r1
        # 'p' neighbours in LBP
        self.p = p
        if self.p <= 8:
            self.map_dtype = np.uint8
        elif self.p <= 16:
            self.map_dtype = np.uint16
        elif self.p <= 32:
            self.map_type = np.uint32
        else:
            raise ValueError('Neighbours p cannot exceed 32')
        # 'w_c', local patch size at center kernel
        if w_center % 2 == 0:
            raise ValueError('Kernel size w_center must be an odd number, but an even number was provided')
        self.w_c = w_center
        if isinstance(w_r1, int):
            w_r1 = [w_r1]
        self.w_r1 = w_r1  # 'w_r1', local patch size for first r' kernel.
        self.r_wr_scales = list(zip_longest(self.r1, self.w_r1, fillvalue=self.w_r1[0]))
        self.riu2_mapping = SharedFunctions.get_riu2_mappings(self.p)
        self.padding = max(r1) + int((max(w_r1) - 1) / 2)
        self.weights = np.fromiter((2 ** i for i in range(self.p)), dtype=self.map_dtype)
        # Compute the angles of separation for neighbour surrounding a point (used in relbp_ni and relbp_rd)
        self.radial_angles = (np.arange(0, self.p) * -(2 * math.pi) / self.p).astype(np.float32)

    def get_outdir(self, noisy_image: bool, scaled_image: bool):
        if noisy_image:
            noise_type = GlobalConfig.get('noise')
            noise_val = GlobalConfig.get('noise_val')
        else:
            noise_type = 'None'
            noise_val = 'None'

        if scaled_image:
            image_scale = int(GlobalConfig.get('test_scale') * 100)
        else:
            image_scale = int(GlobalConfig.get('scale') * 100)

        if GlobalConfig.get('train_noise'):
            return "scale-{}_noise-{}_noiseval-{}_p-{}_wc-{}_r-{}_wr-{}-trainnoise".format(image_scale, noise_type, noise_val,
                                                                                self.p, self.w_c, self.r1, self.w_r1)
        else:
            return "scale-{}_noise-{}_noiseval-{}_p-{}_wc-{}_r-{}_wr-{}".format(image_scale, noise_type, noise_val,
                                                                                self.p, self.w_c, self.r1, self.w_r1)



    def describe(self, image, test_image: bool):
        """
        Perform MRELBP Description for an Image
        :param image: Image object or float32 ndarray image_scaled with zero mean and unit variance. (Use ImageUtils.scale_img first)
        :param test_image: Boolean to determine if we are evaluating the test image
        :return: MRELBP descriptor histogram
        """
        if isinstance(image, DatasetManager.Image):
            if test_image:
                image_data = image.test_data
            else:
                image_data = image.data
        elif isinstance(image, np.ndarray):
            image_data = image
            if self.save_img:
                raise ValueError('save_img set but passed as ndarray instead of DatasetManager.Image')
        else:
            raise ValueError('Invalid image_scaled type')

        # Zero-pad image_scaled with padding border.
        image_padded = pad(array=image_data, pad_width=self.padding, mode='constant', constant_values=0)
        # Allocate memory for output image_scaled
        image_filtered = np.zeros(image_padded.shape, dtype=np.float32)
        # Perform median filter on image_scaled
        SharedFunctions.median_filter(image_padded, self.w_c, self.padding, image_filtered)
        # Make new Image instance to avoid overwriting input image's data

        if self.save_img:
            GenerateExamples.write_image(ImageUtils.convert_float32_image_uint8(image_padded), 'MRELBP', '{}-padded.png'.format(image.name))
            GenerateExamples.write_image(ImageUtils.convert_float32_image_uint8(image_filtered), 'MRELBP', '{}-median-filtered.png'.format(image.name))
            describe_image = DatasetManager.Image(image_filtered, image.name, None)
        else:
            describe_image = DatasetManager.Image(image_filtered, None, None)

        # Return MRELBP descriptor
        return self.calculate_relbp(describe_image)

    def calculate_relbp(self, image):
        """
        Calculates the RELBP descriptor (joint histogram of RELBP_CI, RELBP_NI, RELBP_RD)
        If you apply the Median filter before this, it is the MRELBP descriptor.
        If you perform noise classification + variable filters beforehand, it is the BM3DELBP descriptor.
        :param image: Image object or float32 ndarray, scaled & padded image_scaled of zero mean and unit variance.
        :return: Combined RELBP descriptor histogram
        """
        # Generate r1 and w_r1 parameter pairs depending on whether user passed list or int for each.
        relbp_ni_rd = np.array([], dtype=np.int32)
        for r, w_r in self.r_wr_scales:
            if w_r % 2 == 0:
                raise ValueError('Kernel size w_r1 must be an odd number, but an even number was provided')
            if isinstance(image, np.ndarray):
                relbp_ni, relbp_rd = self.relbp_ni_rd(image, r, w_r)
            else:
                relbp_ni, relbp_rd = self.relbp_ni_rd(image.data, r, w_r)

            if self.save_img:
                if isinstance(image, np.ndarray):
                    raise ValueError('save_img set but passed as ndarray instead of DatasetManager.Image')
                else:
                    GenerateExamples.write_image(ImageUtils.convert_float32_image_uint8(relbp_ni), 'MRELBP', '{}-RELBPNI_r-{}_wr-{}.png'.format(image.name, r, w_r))
                    GenerateExamples.write_image(ImageUtils.convert_float32_image_uint8(relbp_rd), 'MRELBP', '{}-RELBPRD_r-{}_wr-{}.png'.format(image.name, r, w_r))

            relbp_ni_hist = np.histogram(relbp_ni, self.p + 2)[0].astype(dtype=np.int32)
            relbp_rd_hist = np.histogram(relbp_rd, self.p + 2)[0].astype(dtype=np.int32)
            relbp_ni_rd = np.concatenate((relbp_ni_rd, relbp_ni_hist, relbp_rd_hist))

        if isinstance(image, np.ndarray):
            relbp_ci_hist = self.relbp_ci(image, self.w_c, self.padding)
        else:
            relbp_ci_hist = self.relbp_ci(image.data, self.w_c, self.padding)

        combined_histogram = np.concatenate((relbp_ci_hist, relbp_ni_rd))
        return combined_histogram

    @staticmethod
    @nb.jit(nb.int32[:](nb.float32[:, :], nb.uint16, nb.uint16))
    def relbp_ci(image, w_c, padding):
        """

        :param image: float32 ndarray, scaled, padded image_scaled of zero mean and unit variance
                    For MRELBP_CI descriptor, also apply median filter beforehand
        :param w_c: CI kernel size
        :param padding: padding of image_scaled
        :return: RELBP_CI histogram
        """
        width, height = image.shape
        x_centre, y_centre = math.ceil(width / 2), math.ceil(height / 2)
        patch = int((w_c - 1) / 2)

        # Get the image_scaled excluding the zero-padding
        image_no_pad = image[padding:height - padding - 1, padding:width - padding - 1]
        # Get the central w_c*w_c section
        centre = image[x_centre - patch:x_centre + patch + 1, y_centre - patch:y_centre + patch + 1]

        # Calculate Centre Histogram
        diffs = centre - np.mean(image_no_pad)
        centre_hist = np.array([np.sum(diffs >= 0), np.sum(diffs < 0)], dtype=np.int32)

        return centre_hist

    def relbp_ni_rd(self, image: np.ndarray, r1, w_r1, r2=None, w_r2=None):
        """
        Calculate RELBP_NI and RELBP_RD descriptor
        RELBP_NI Uses mean neighbourhood pixel intensity to threshold the neighbourhood to generate the binary pattern
        RELBP_RD Uses the larger neighbourhood thresholded against the nearer neighbourhood's mean pixel intensities
        :param image: float32 ndarray, scaled, padded image_scaled of zero mean and unit variance.
                    For MRELBP_NI, RD descriptors, also apply median filter beforehand
        :param r1: Radius for larger neighbourhood
        :param w_r1: Kernel size for larger neighbourhood
        :param r2: Optional: Radius for smaller neighbourhood
        :param w_r2: Optional: Kernel size for smaller neighbourhood
        :return: RELBP_NI, RELBP_RD histogram descriptors
        """
        # Handle initialisation of optional arguments
        if r2 is None:
            # This is the initialisation defined in the paper's section (9)
            r2 = r1 - 1
        else:
            if r1 <= r2:
                raise ValueError('r2 argument should be smaller or equal to r1')

        if w_r2 is None:
            w_r2 = w_r1 - 2
        else:
            if w_r2 % 2 == 0:
                raise ValueError('w_r2 kernel size must be odd value, but is even')
            if w_r1 <= w_r2:
                raise ValueError('w_r2 argument should be smaller or equal to r1')
        r1_offset = int((w_r1 - 1) / 2)
        r2_offset = int((w_r2 - 1) / 2)

        # Amount to offset from the patch's centre pixel to capture the w*w sized patch
        width, height = image.shape

        # Empty ndarray to store mapped values LBP values for each pixel
        LBPNI_mapped = np.zeros((width - 2 * self.padding, height - 2 * self.padding), dtype=self.map_dtype)
        LBPRD_mapped = np.zeros((width - 2 * self.padding, height - 2 * self.padding), dtype=self.map_dtype)

        lbp_ni = np.zeros((width, height), dtype=np.uint32)
        lbp_rd = np.zeros((width, height), dtype=np.uint32)
        try:
            self.perform_ni_rd_thresholding(image, self.padding, r1, r2, r1_offset, r2_offset,
                                            self.radial_angles, self.weights, lbp_ni, lbp_rd)
        except RuntimeWarning as e:
            print(e)
            print("RuntimeWarning in perform_ni_rd happened with the following arguments:")
            print("image type:", type(image), image.dtype)
            print("padding:", self.padding)
            print("r1", r1)
            print("r2", r2)
            print("r1_offset", r1_offset)
            print("r2_offset", r2_offset)
            print("radial angles", self.radial_angles)
            print("weights", self.weights)
            print("lbp_ni", lbp_ni)
            print("lbp_rd", lbp_rd)


        # Trim extra rows and columns
        lbp_ni = lbp_ni[:width-2 * self.padding, :height-2 * self.padding]
        lbp_rd = lbp_rd[:width-2 * self.padding, :height-2 * self.padding]

        for x in range(width - 2 * self.padding):
            for y in range(height - 2 * self.padding):
                LBPNI_mapped[x][y] = self.riu2_mapping[lbp_ni[x][y]]
                LBPRD_mapped[x][y] = self.riu2_mapping[lbp_rd[x][y]]

        return LBPNI_mapped, LBPRD_mapped

    @staticmethod
    @nb.guvectorize([(nb.float32[:, :], nb.uint16, nb.uint16, nb.uint16, nb.uint16, nb.uint16, nb.float32[:], nb.uint16[:], nb.uint32[:, :], nb.uint32[:, :])], '(x,y),(),(),(),(),(),(n),(n)->(x,y),(x,y)')
    def perform_ni_rd_thresholding(image, padding, r1, r2, r1_offset, r2_offset, radial_angles, weights, lbp_ni, lbp_rd):
        """
        Vectorized function to perform RELBP_NI and RELBP_RD thresholding
        :param image: Greyscale image_scaled
        :param padding: Padding used on image_scaled
        :param r1: Radius for larger neighbourhood
        :param r2: Radius for smaller neighbourhood
        :param r1_offset: Pixel offset to achieve r1 kernel size
        :param r2_offset: Pixel offset to achieve r2 kernel size
        :param radial_angles: Angles for each of the neighbouring points from the evaluated point
        :param weights: 2^n weights for calculating the LBP_RD and RELBP_NI values (not riu2 mapped)
        :param lbp_ni: LBP_NI values evaluated for each pixel
        :param lbp_rd: LBP_RD values evaluated for each pixel
        :return:
        """
        width, height = image.shape
        for x_c in np.arange(padding, width - padding, dtype=np.uint16):
            for y_c in np.arange(padding, height - padding, dtype=np.uint16):
                x1s = np.empty(len(radial_angles), dtype=np.float32)
                y1s = np.empty(len(radial_angles), dtype=np.float32)
                x2s = np.empty(len(radial_angles), dtype=np.float32)
                y2s = np.empty(len(radial_angles), dtype=np.float32)

                for i in range(len(radial_angles)):
                    x1s[i] = x_c + r1 * math.cos(radial_angles[i])
                    y1s[i] = y_c + r1 * math.sin(radial_angles[i])
                    x2s[i] = x_c + r2 * math.sin(radial_angles[i])
                    y2s[i] = y_c + r2 * math.sin(radial_angles[i])

                N_vals_r1 = np.zeros(len(radial_angles), dtype=np.float32)
                N_vals_r2 = np.zeros(len(radial_angles), dtype=np.float32)
                SharedFunctions.get_radial_means(image, x1s, y1s, r1_offset, N_vals_r1)
                SharedFunctions.get_radial_means(image, x2s, y2s, r2_offset, N_vals_r2)

                # Neighbourhood mean thresholding (NI descriptor)
                N_vals_r1 = N_vals_r1 - np.mean(N_vals_r1)
                N_thresholded_ni = N_vals_r1 >= 0
                lbp_ni[x_c-padding, y_c-padding] = np.sum(N_thresholded_ni * weights)

                # Radial Neighbourhood thresholding (RD descriptor)
                N_vals_r2 = N_vals_r1 - N_vals_r2
                N_thresholded_rd = N_vals_r2 >= 0
                lbp_rd[x_c-padding, y_c-padding] = np.sum(N_thresholded_rd * weights)


class MedianRobustExtendedLBPPredictor(ImageClassifierInterface):

    def begin_cross_validation(self) -> Tuple[List[np.array], List[str]]:
        return super().begin_cross_validation()

    def __init__(self, dataset: List[DatasetManager.Image], cross_validator):
        super().__init__(dataset, cross_validator)
        self.classifier = None

    def train(self, train: List[DatasetManager.Image]):
        X_train = [img.featurevector for img in train]
        # Convert List[narray] to ndarray
        X_train = np.stack(X_train, axis=0)
        X_train = X_train.astype(np.float64)
        y_train = [img.label for img in train]

        #self.noise_classifier = SVC(C=1000000, gamma=0.01)
        self.classifier = SVC()
        self.classifier.fit(X_train, y_train)

    def classify(self, X) -> List[str]:
        y_predictions = self.classifier.predict(X)
        return y_predictions