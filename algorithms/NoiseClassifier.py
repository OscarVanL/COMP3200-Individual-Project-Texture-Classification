from typing import List

from collections import Counter
from algorithms import SharedFunctions
from algorithms.AlgorithmInterfaces import ImageProcessorInterface, NoiseClassifierInterface
from scipy.stats import skew, kurtosis
from sklearn.svm import SVC
from skimage.util import pad
from config import GlobalConfig
from example import GenerateExamples
from data import ImageUtils, DatasetManager
import numpy as np
import os


class NoiseClassifier(ImageProcessorInterface):
    """
    Implementation of the Noise Classifier used in the BM3DELBP and SARBM3DELBP paper.
    This uses a modified version of the noise_classifier from: https://ieeexplore.ieee.org/document/4488699
    The principle of this paper is as follows:
    1 Take a noisy image with unknown noise type
    2 Apply the 3 noise filter types, BM3D filter, Homomorphic filter, Median filter to get 3 estimates
        for the non-noisy image
    3 Subtract the filtered images from the noisy image to get 3 noise estimates.
    4 Calculate the kurtosis (Kurt) and skewness (Skew) on these noise estimates.
    5 Use a minimum distance pattern noise_classifier to measure the similarity of Kurt and Skew to these measured on noise
        samples.
        IE: Is one of the noise estimates very similar to a sample of a type of noise? Then it probably is that type.

    """

    def __init__(self, save_img=False):
        super().__init__(save_img)

    def get_outdir(self, noisy_image: bool, scaled_image: bool):
        if GlobalConfig.get('train_noise'):
            return "scale-{}-trainnoise".format(int(GlobalConfig.get('scale') * 100))
        else:
            return "scale-{}".format(int(GlobalConfig.get('scale') * 100))

    def describe(self, image, test_image: bool, sigma_psd=70/255, cutoff_freq=10, a=0.75, b=1.0):
        if isinstance(image, DatasetManager.Image):
            if test_image:
                image_data = image.test_data
            else:
                image_data = image.data
        else:
            image_data = image.copy()

        # Perform BM3D Filter
        image_bm3d_filtered = SharedFunctions.bm3d_filter(image_data, 70/255)
        # Perform Homomorphic filter, Note: This requires images to be normalised in range [0, 255]
        image_scaled_255 = ImageUtils.convert_float32_image_uint8(image_data)
        cutoff, a, b = 10, 0.75, 0.1
        image_homomorphic_filtered = SharedFunctions.homomorphic_filter(image_scaled_255, cutoff, a, b)
        image_homomorphic_filtered = ImageUtils.convert_uint8_image_float32(image_homomorphic_filtered)
        # Perform Median filter. Padding is required for median filter.
        image_padded = pad(array=image_data, pad_width=1, mode='constant', constant_values=0)
        image_median_filtered = np.zeros(image_padded.shape, dtype=np.float32)
        SharedFunctions.median_filter(image_padded, 3, 1, image_median_filtered)
        image_median_filtered = image_median_filtered[1:-1, 1:-1]  # Remove padding now median filter done

        # Subtract original image from filtered image to get noise only
        bm3d_noise = image_data - image_bm3d_filtered
        homomorphic_noise = image_data - image_homomorphic_filtered
        median_noise = image_data - image_median_filtered

        if self.save_img:
            if isinstance(image, DatasetManager.Image):
                GenerateExamples.write_image(ImageUtils.convert_float32_image_uint8(image_bm3d_filtered),
                                             os.path.join('BM3DELBP', 'NoiseClassifier', 'Filtered Images'),
                                             '{}-{}-{}-BM3D-filtered.png'.format(image.name, image.test_noise,
                                                                                 image.test_noise_val))
                GenerateExamples.write_image(ImageUtils.convert_float32_image_uint8(image_homomorphic_filtered),
                                             os.path.join('BM3DELBP', 'NoiseClassifier', 'Filtered Images'),
                                             '{}-{}-{}-homomorphic-filtered-cutoff_{}-a_{}-b_{}.png'.format(image.name,
                                                                                                            image.test_noise,
                                                                                                            image.test_noise_val,
                                                                                                            cutoff, a,
                                                                                                            b))
                GenerateExamples.write_image(ImageUtils.convert_float32_image_uint8(image_median_filtered),
                                             os.path.join('BM3DELBP', 'NoiseClassifier', 'Filtered Images'),
                                             '{}-{}-{}-median-filtered.png'.format(image.name, image.test_noise,
                                                                                   image.test_noise_val))
                GenerateExamples.write_image(ImageUtils.convert_float32_image_uint8(bm3d_noise),
                                             os.path.join('BM3DELBP', 'NoiseClassifier', 'Noise Estimates'),
                                             '{}-{}-{}-BM3D-noise-estimate.png'.format(image.name, image.test_noise,
                                                                                       image.test_noise_val))
                GenerateExamples.write_image(ImageUtils.convert_float32_image_uint8(homomorphic_noise),
                                             os.path.join('BM3DELBP', 'NoiseClassifier', 'Noise Estimates'),
                                             '{}-{}-{}-homomorphic-filtered-cutoff_{}-a_{}-b_{}.png'.format(image.name,
                                                                                                            image.test_noise,
                                                                                                            image.test_noise_val,
                                                                                                            cutoff, a,
                                                                                                            b))
                GenerateExamples.write_image(ImageUtils.convert_float32_image_uint8(median_noise),
                                             os.path.join('BM3DELBP', 'NoiseClassifier', 'Noise Estimates'),
                                             '{}-{}-{}-median-noise-estimate.png'.format(image.name, image.test_noise,
                                                                                         image.test_noise_val))
            else:
                raise ValueError('save_img set but not passed as DatasetManager.Image or BM3DELBPImage')

        kurtosis_bm3d = kurtosis(a=bm3d_noise, axis=None, fisher=False, nan_policy='raise')
        skewness_bm3d = skew(a=bm3d_noise, axis=None, nan_policy='raise')
        kurtosis_homomorphic = kurtosis(a=homomorphic_noise, axis=None, fisher=False, nan_policy='raise')
        skewness_homomorphic = skew(a=homomorphic_noise, axis=None, nan_policy='raise')
        kurtosis_median = kurtosis(a=median_noise, axis=None, fisher=False, nan_policy='raise')
        skewness_median = skew(a=median_noise, axis=None, nan_policy='raise')

        if GlobalConfig.get('debug'):
            print("BM3D Filtered Kurtosis:", kurtosis_bm3d, ", Skewness: ", skewness_bm3d)
            print("Homomorphic Filtered Kurtosis:", kurtosis_homomorphic, ", Skewness: ", skewness_homomorphic)
            print("Median Filtered Kurtosis:", kurtosis_median, ", Skewness: ", skewness_median)

        # Generate image featurevector of 6 characteristics
        featurevector = np.array([kurtosis_bm3d, skewness_bm3d,
                                  kurtosis_homomorphic, skewness_homomorphic,
                                  kurtosis_median, skewness_median])
        return featurevector


class NoiseTypePredictor(NoiseClassifierInterface):
    """
    Train the Noise Type predictor. This works differently to the predictors for other algorithms in this project.
    To summarise the approach in the BM3DELBP paper...
    For each cross-validation of BM3DELBP:
        * During the training phase, only images from BM3DELBP's training set are used.
        * Each training image has each type of noise added, forming a 3x bigger training set.
        * A featurevector is generated for each image as in NoiseClassifier.describe()
        * Finally, a SVM noise_classifier is chosen with a polynomial kernel to identify the type of noise in the images.
        * This can be validated against the BM3DELBP test set.
    """

    def __init__(self, dataset: List[DatasetManager.Image], cross_validator):
        super().__init__(dataset, cross_validator)
        self.classifier = None

    # def __init__(self, X_dataset, y_dataset, cross_validator):
    #     #super().__init__(dataset, cross_validator)
    #     self.X_dataset = X_dataset
    #     self.y_dataset = y_dataset
    #     self.classifier = None

    def begin_cross_validation(self):
        """
        Note: This cross validator is only used for benchmarking the Noise Classifier. In practise BM3DELBP will
        pass in the Train values on instantiation and train() will be called.
        :return: Returns the test label and predicted label for every fold.
        """
        fold = 1
        test_y_all = []
        pred_y_all = []
        for train_index, test_index in self.cross_validator.split(self.dataset_X, self.dataset_y):


            print("Performing fold", fold)
            # Remove any existing classifiers
            self.classifier = None

            self.classifier = NoiseTypePredictor(self.dataset, None)
            # Train noise noise_classifier on this fold
            noise_train_X = [self.dataset_X[index] for index in train_index]
            noise_train_y = [self.dataset_y[index] for index in train_index]

            print("DEBUG: Class Counter (to check if train classes are balanced)")
            print(Counter(noise_train_y))

            self.train_on(noise_train_X, noise_train_y)

            # Now test noise noise_classifier
            noise_test_X = [self.dataset_X[index] for index in test_index]
            noise_test_y = [self.dataset_y[index] for index in test_index]
            noise_pred_y = self.classify(noise_test_X)

            test_y_all.extend(noise_test_y)
            pred_y_all.extend(noise_pred_y)
            fold += 1

        return test_y_all, pred_y_all

    def train(self):
        # Convert List[narray] to ndarray
        X = np.stack(self.dataset_X, axis=0)
        self.classifier = SVC()
        self.classifier.fit(X, self.dataset_y)

    def train_on(self, X: List[np.ndarray], y: List[str]):
        # Convert List[narray] to ndarray
        X = np.stack(X, axis=0)
        self.classifier = SVC()
        self.classifier.fit(X, y)

    def classify(self, X) -> List[str]:
        y_predictions = self.classifier.predict(X)
        return y_predictions
