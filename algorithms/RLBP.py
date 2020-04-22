from typing import Tuple, List

from sklearn.svm import SVC
from skimage import feature
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from algorithms.AlgorithmInterfaces import ImageProcessorInterface, ImageClassifierInterface
from data import DatasetManager
from example import GenerateExamples
from algorithms import SharedFunctions
from itertools import zip_longest
from config import GlobalConfig

"""
Implementation of Robust LBP (RLBP) and Multiresolution LBP (MRLBP) paper using Skimage
RLBP uses a fixed number of points and r, but introduces riu2 uniform patterns.
RLBP: https://ieeexplore.ieee.org/abstract/document/903698
MRLBP introduces multi-resolution sampling, also using riu2 patterns.
MRLBP: https://ieeexplore.ieee.org/document/1017623
"""


class RobustLBP(ImageProcessorInterface):
    """
    RLBP is a very simple implementation using a r of 1 and 8 points.
    It does not have any configurable parameters.
    """

    def __init__(self, save_img=False):
        super().__init__(save_img)
        self.riu2_mapping = SharedFunctions.get_riu2_mappings(8)

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

        return "scale-{}_noise-{}_noiseval-{}".format(image_scale, noise_type, noise_val)

    def describe(self, image, test_image: bool):
        if isinstance(image, DatasetManager.Image):
            if test_image:
                image_data = image.test_data
            else:
                image_data = image.data
        elif isinstance(image, np.ndarray):
            image_data = image
            if self.save_img:
                raise ValueError('save_img set but passed ndarray instead of DatasetManager.Image')
        else:
            raise ValueError('Invalid image argument type')

        rlbp = feature.local_binary_pattern(image_data, P=8, R=1, method='uniform')
        if self.save_img:
            # Generate the image_scaled without uniform mappings for illustrative purposes.
            rlbp_out = feature.local_binary_pattern(image_data, P=8, R=1, method='default').astype(np.uint8)
            GenerateExamples.write_image(rlbp_out, 'RLBP', image.name + ".png")

        hist = np.histogram(rlbp, bins=9)[0]

        # return the histogram of Local Binary Patterns
        return hist


class RobustLBPPredictor(ImageClassifierInterface):
    """
    A Machine Learning predictor for RLBP descriptors.
    """

    # Todo: Implement predictor
    def __init__(self, dataset: List[DatasetManager.Image], cross_validator):
        super().__init__(dataset, cross_validator)

    def begin_cross_validation(self) -> Tuple[List[np.array], List[str]]:
        return super().begin_cross_validation()

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


class MultiresolutionLBP(ImageProcessorInterface):
    """
    MRLBP is an extension of RLBP that introduces a variable number of points and radii, the concatenation of multiple
    scales histograms gives more discriminative histograms that should allow for scale invariance.

    In https://ieeexplore.ieee.org/document/1017623, the P,R values with best performance when training on a single
    rotation angle to classify across many angles was (8,1)+(16,2)+(24,3).
    """

    def __init__(self, p=None, r=None, save_img=False):
        """
        :param p: int or [int]. Number of neighbours for MRLBP Length must match r
        :param r: int or [int]. Radius(s) to to use for MRLBP. Length must match p
        :param save_img:
        """
        super().__init__(save_img)
        if p is None:
            p = [8, 16, 24]
        if r is None:
            r = [1, 2, 3]
        if isinstance(p, int):
            p = [p]
        if isinstance(r, int):
            r = [r]
        self.p = p  # P
        self.r = r  # R
        self.p_r_scales = list(zip_longest(self.p, self.r, fillvalue=self.r[0]))

    def get_outdir(self, noisy_image: bool, scaled_image: bool):
        """
        Gets a string name for the read/write directory depending on the configuration
        :param noisy_image: Whether the algorithm is being applied to images containing noise
        :param scaled_image: Whether the algorithm is being applied to a scaled image
        :return: String output directory name
        """
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

        return "scale-{}_noise-{}_noiseval-{}_p-{}_r-{}".format(image_scale, noise_type, noise_val, self.p, self.r)

    def describe(self, image, test_image: bool):
        if isinstance(image, DatasetManager.Image):
            if test_image:
                image_data = image.test_data
            else:
                image_data = image.data
        elif isinstance(image, np.ndarray):
            image_data = image
        else:
            raise ValueError('Invalid image argument type')

        combined_histogram = np.array([], dtype=np.int32)
        for p, r in self.p_r_scales:
            mrlbp = feature.local_binary_pattern(image_data, p, r, method="uniform")
            mrlbp_hist = np.histogram(mrlbp, p + 1)[0].astype(dtype=np.int32)
            combined_histogram = np.concatenate((combined_histogram, mrlbp_hist))
            if self.save_img:
                # Generate the image_scaled without uniform mappings for illustrative purposes.
                mrlbp_out = feature.local_binary_pattern(image.data, P=p, R=r, method='default').astype(np.uint8)
                if isinstance(image, np.ndarray):
                    raise ValueError('save_img set but passed ndarray instead of DatasetManager.Image')
                else:
                    GenerateExamples.write_image(mrlbp_out, 'MRLBP', '{}_p-{}_r-{}.png'.format(image.name, p, r))

        # # normalize the histogram
        # hist = hist.astype("float")
        # hist /= hist.sum()

        # return the histogram of Local Binary Patterns
        return combined_histogram


class MultiresolutionLBPPredictor(ImageClassifierInterface):
    """
    A Machine Learning predictor for MRLBP descriptors.
    The parameters used for the model are the same as those in https://ieeexplore.ieee.org/document/1017623
    This uses 3-NN with Mahalanobis distance
    """

    def __init__(self, dataset: List[DatasetManager.Image], cross_validator, classifier_type):
        """
        :param dataset: Dataset to train/test with
        :param cross_validator: Cross validator to use for train/test splits
        :param classifier_type: Whether to use 'svm' or 'knn' for the noise_classifier
        """
        super().__init__(dataset, cross_validator)
        self.classifier_type = classifier_type
        self.classifier = None

    def begin_cross_validation(self) -> Tuple[List[np.array], List[str]]:
        print("Training MRLBP using", self.classifier_type, "noise_classifier.")
        return super().begin_cross_validation()

    def train(self, train: List[DatasetManager.Image]):
        X_train = [img.featurevector for img in train]
        # Convert List[narray] to ndarray
        X_train = np.stack(X_train, axis=0)
        X_train = X_train.astype(np.float64)
        y_train = [img.label for img in train]

        if self.classifier_type == 'knn':
            # Calculate transposed covariance matrix as described here: https://stackoverflow.com/a/55623162/6008271
            X_train_cov = np.linalg.inv(np.cov(X_train.transpose())).transpose()


            self.classifier = KNeighborsClassifier(n_neighbors=3,
                                                   algorithm='brute',
                                                   metric='mahalanobis',
                                                   metric_params={'VI': X_train_cov})
            self.classifier.fit(X_train, y_train)
        elif self.classifier_type == 'svm':
            self.classifier = SVC()
            self.classifier.fit(X_train, y_train)
        else:
            raise ValueError('Invalid Classifier Type specified for MultiresolutionLBPPredictor')

    def classify(self, X: List[np.array]):
        y_predictions = self.classifier.predict(X)
        return y_predictions
