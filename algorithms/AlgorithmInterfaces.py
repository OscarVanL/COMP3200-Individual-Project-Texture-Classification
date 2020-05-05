from abc import abstractmethod

from config import GlobalConfig
from data import DatasetManager
import numpy as np
from typing import Tuple, List


class ImageProcessorInterface:
    """
    Interface used for all algorithm's featurevector descriptions
    """

    @abstractmethod
    def __init__(self, save_img=False):
        self.save_img = save_img  # Whether to save images during runs of MRELBP for illustrative purposes.

    @abstractmethod
    def get_outdir(self, noisy_image: bool, scaled_image: bool):
        """
        Get the folder name for outputting the serialised featurevectors.
        :param noisy_image: Whether the featurevectors were generated on a noisy image, or an ordinary image
        :param scaled_image: Whether the featurevectors were generated on a scaled image
        :return: Output directory name
        """

    @abstractmethod
    def describe(self, image, test_image: bool):
        """
        Describe a DatasetManager.Image or np.ndarray with some sort of featurevector
        :param image: Input DatasetManager.Image or np.ndarray
        :param test_image: Whether the test image should be described
        :return: Featurevector (algorithm specific)
        """
        pass


class ImageClassifierInterface:
    """
    Interface used for all algorithm's classifiers
    """

    @abstractmethod
    def __init__(self, dataset: List[DatasetManager.Image], cross_validator):
        self.dataset = dataset
        self.dataset_y = [image.label for image in dataset]
        self.cross_validator = cross_validator
        self.classifier = None

    @abstractmethod
    def begin_cross_validation(self) -> Tuple[List[np.array], List[str]]:
        """
        Begins cross validation by taking a ShuffleSplit fold of the dataset, training and then testing.
        Repeats for the n_splits configured in the cross validator.
        :return: Returns the test label and predicted label for every fold.
        """
        fold = 1
        test_X_all = []
        test_y_all = []
        pred_y_all = []
        for train_index, test_index in self.cross_validator.split(self.dataset, self.dataset_y):
            print("Performing fold", fold)
            # Remove any existing noise_classifier
            self.classifier = None
            # Train on this fold
            train = [self.dataset[index] for index in train_index]
            self.train(train)

            # Get X and y values for testing on this fold
            test_X = []
            test_y = []
            test_image_exists = (GlobalConfig.get('noise') is not None) or (GlobalConfig.get('test_scale') is not None)
            if GlobalConfig.get('rotate'):
                for index in test_index:
                    # Get featurevectors and labels for each rotation of the image
                    for rotation in self.dataset[index].test_rotations:
                        if test_image_exists:
                            test_X.append(rotation.test_featurevector)
                            test_y.append(rotation.label)
                        else:
                            if rotation.featurevector is None:
                                raise ValueError('Featurevector not assigned for rotation')
                            test_X.append(rotation.featurevector)
                            test_y.append(rotation.label)

            else:
                if test_image_exists:
                    # If we have a test featurevector, make sure we test on this.
                    test_X = [self.dataset[index].test_featurevector for index in test_index]
                else:
                    test_X = [self.dataset[index].featurevector for index in test_index]
                test_y = [self.dataset[index].label for index in test_index]

            # Classify this fold.
            pred_y = self.classify(test_X)
            test_X_all.extend(test_X)
            test_y_all.extend(test_y)
            pred_y_all.extend(pred_y)

            fold += 1

        return test_y_all, pred_y_all

    @abstractmethod
    def train(self, train: List[DatasetManager.Image]):
        """
        Train the Image Classifier on a List[DatasetManager.Image] with calculated featurevectors
        :return: None
        """

    @abstractmethod
    def classify(self, X) -> List[str]:
        """
        Performs classification with the trained model upon unseen featurevectors
        :param X: test_X values to make predictions for
        :return: List of class predictions
        """


class NoiseClassifierInterface:
    """
    Interface used for BM3DELBP's noise noise_classifier
    """

    @abstractmethod
    def __init__(self, dataset, cross_validator):
        self.cross_validator = cross_validator

        self.dataset_X = []
        self.dataset_y = []
        for image in dataset:
            self.__add_to_dataset__(image)


    def __add_to_dataset__(self, image):
        # Add no-noise twice so that each class is equal in size.
        if GlobalConfig.get('noise') is None:
            self.dataset_X.append(image.no_noise_featurevector)
            self.dataset_y.append('no-noise')
            if not GlobalConfig.get('rotate'):
                self.dataset_X.append(image.no_noise_featurevector)
                self.dataset_y.append('no-noise')
        if not GlobalConfig.get('rotate'):
            self.dataset_X.append(image.gauss_25_noise_featurevector)
            self.dataset_y.append('gaussian')
            self.dataset_X.append(image.speckle_004_noise_featurevector)
            self.dataset_y.append('speckle')
            self.dataset_X.append(image.salt_pepper_004_noise_featurevector)
            self.dataset_y.append('salt-pepper')
        self.dataset_X.append(image.gauss_10_noise_featurevector)
        self.dataset_y.append('gaussian')
        self.dataset_X.append(image.speckle_002_noise_featurevector)
        self.dataset_y.append('speckle')
        self.dataset_X.append(image.salt_pepper_002_noise_featurevector)
        self.dataset_y.append('salt-pepper')


        # Ensure noise featurevectors for rotations of images are added too
        if image.test_rotations is not None:
            for image in image.test_rotations:
                self.__add_to_dataset__(image)


    @abstractmethod
    def begin_cross_validation(self) -> Tuple[List[np.array], List[str]]:
        pass

    def train(self):
        """
        Trains the Image Classifier on the whole dataset passed to the Noise Classifier at instantiation
        :return: None
        """

    @abstractmethod
    def train_on(self, X: List[np.ndarray], y: List[str]):
        """
        Train the Image Classifier on a list of calculated noise featurevectors
        :return: None
        """

    @abstractmethod
    def classify(self, X) -> List[str]:
        """
        Performs classification with the trained model on unseen noise featurevectors
        :param X: test_X values to make predictions for
        :return: List of class predictions
        """