from typing import List, Tuple

from algorithms import SharedFunctions, SARBM3D, MRELBP
from algorithms.AlgorithmInterfaces import ImageProcessorInterface, ImageClassifierInterface
from algorithms.NoiseClassifier import NoiseTypePredictor
from config import GlobalConfig
from skimage.util import pad
import numpy as np
from data import ImageUtils, DatasetManager


class BM3DELBP(ImageProcessorInterface):
    """
    Implementation of BM3DELBP and SARBM3DELBP adaptive noise type detection & filtering method.
    From Paper: An Improved Feature Extraction Method for Texture Classification with Increased Noise Robustness
    https://ieeexplore.ieee.org/document/8902765
    """

    def __init__(self, save_img=False):
        self.elbp = MRELBP.MedianRobustExtendedLBP()
        super().__init__(save_img)

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

        return "scale-{}_noise={}_noiseval-{}".format(image_scale, noise_type, noise_val)

    def describe(self, image, test_image: bool):
        """
        Essentially BM3DELBP's descriptors are the same as that in MRELBP!
        The primary difference is the noise identification + filtering stage.
        :param image: Image to classify
        :param test_image: Whether this is a test image with a noise type applied
        :return:
        """

        if self.save_img:
            # Todo: Implement example image generation for report
            pass
        return self.elbp.calculate_relbp(image)


class BM3DELBPImage(DatasetManager.Image):
    def __init__(self, image: DatasetManager.Image):
        """
        A modified version of DatasetManger.Image class specific to the BM3DELBP implementation.
        This takes an Image, applies each noise type, and for each image generates the featurevector for the NoiseClassifier.
        Doing this for each fold would be enormously slow, so it allows this to be reused.
        """
        super().__init__(image.data, image.name, image.label)

        self.gauss_10_data = None
        self.gauss_10_noise_featurevector = None  # Feaurevector for the noise identifier
        self.gauss_10_prediction = None  # When the Noise Classifier has predicted the noise type, store it here
        self.gauss_10_bm3d_featurevector = None

        self.gauss_25_data = None
        self.gauss_25_noise_featurevector = None
        self.gauss_25_prediction = None
        self.gauss_25_bm3d_featurevector = None

        self.speckle_002_data = None
        self.speckle_002_noise_featurevector = None
        self.speckle_002_prediction = None
        self.speckle_002_bm3d_featurevector = None

        self.salt_pepper_002_data = None
        self.salt_pepper_002_noise_featurevector = None
        self.salt_pepper_002_prediction = None
        self.salt_pepper_002_bm3d_featurevector = None

    def generate_gauss_10(self, noise_classifier):
        self.gauss_10_data = ImageUtils.add_gaussian_noise_skimage(self.data, 10)
        self.gauss_10_noise_featurevector = noise_classifier.describe(self.gauss_10_data, test_image=True)

    def generate_gauss_25(self, noise_classifier):
        self.gauss_25_data = ImageUtils.add_gaussian_noise_skimage(self.data, 25)
        self.gauss_25_noise_featurevector = noise_classifier.describe(self.gauss_25_data, test_image=True)

    def generate_speckle_002(self, noise_classifier):
        self.speckle_002_data = ImageUtils.add_speckle_noise_skimage(self.data, 0.02)
        self.speckle_002_noise_featurevector = noise_classifier.describe(self.speckle_002_data, test_image=True)

    def generate_salt_pepper_002(self, noise_classifier):
        self.salt_pepper_002_data = ImageUtils.add_salt_pepper_noise_skimage(self.data, 0.02)
        self.salt_pepper_002_noise_featurevector = noise_classifier.describe(self.salt_pepper_002_data,
                                                                                  test_image=True)


class BM3DELBPPredictor(ImageClassifierInterface):

    def __init__(self, dataset: List[BM3DELBPImage], cross_validator):
        super().__init__(None, cross_validator)
        self.elbp = MRELBP.MedianRobustExtendedLBP()
        self.dataset = dataset
        self.classifier = None
        self.noise_classifier = None

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
            # Train NoiseClassifier on Train indexes only
            noise_classifier_train = [self.dataset[index] for index in train_index]
            self.noise_classifier = NoiseTypePredictor(noise_classifier_train, None)
            self.noise_classifier.train()

            # Establish connection to MATLAB Engine for SAR-BM3D filter
            self.sar_bm3d = SARBM3D.SARBM3DFilter()

            for index in test_index:
                # Do NoiseClassifier Classification
                image = self.dataset[index]
                image.gauss_10_prediction = self.noise_classifier.classify(image.gauss_10_noise_featurevector)
                image.gauss_25_prediction = self.noise_classifier.classify(image.gauss_25_noise_featurevector)
                image.speckle_002_prediction = self.noise_classifier.classify(image.speckle_002_noise_featurevector)
                image.salt_pepper_002_prediction = self.noise_classifier.classify(
                    image.salt_pepper_002_noise_featurevector)

                # Apply relevant filter (BM3D for Gaussian, SAR-BM3D for speckle, Median for salt-pepper.)
                image.gauss_10_data = self.apply_filter(image.gauss_10_data, image.name, image.gauss_10_prediction)
                image.gauss_25_data = self.apply_filter(image.gauss_25_data, image.name, image.gauss_25_prediction)
                image.speckle_002_data = self.apply_filter(image.speckle_002_data, image.name,
                                                           image.speckle_002_prediction)
                image.salt_pepper_002_data = self.apply_filter(image.salt_pepper_002_data, image.name,
                                                               image.salt_pepper_002_prediction)

                # Generate BM3DELBP descriptor for each image
                image.gauss_10_bm3d_featurevector = self.elbp.calculate_relbp(image.gauss_10_data)
                image.gauss_25_bm3d_featurevector = self.elbp.calculate_relbp(image.gauss_25_data)
                image.speckle_002_bm3d_featurevector = self.elbp.calculate_relbp(image.speckle_002_data)
                image.salt_pepper_002_bm3d_featurevector = self.elbp.calculate_relbp(image.salt_pepper_002_data)

                test_X_all.append(image.gauss_10_bm3d_featurevector)
                test_y_all.append(image.label)
                test_X_all.append(image.gauss_25_bm3d_featurevector)
                test_y_all.append(image.label)
                test_X_all.append(image.speckle_002_bm3d_featurevector)
                test_y_all.append(image.label)
                test_X_all.append(image.salt_pepper_002_bm3d_featurevector)

            # Disconnect from MATLAB Python Engine
            self.sar_bm3d.disconnect_matlab()

            fold += 1

        return test_y_all, pred_y_all

    def apply_filter(self, image_data, image_name, noise_prediction):
        if noise_prediction == 'gaussian':
            # Apply BM3D filter
            image_filtered = SharedFunctions.bm3d_filter(image_data)
        elif noise_prediction == 'speckle':
            # Apply SAR-BM3D filter
            image_filtered = self.sar_bm3d.sar_bm3d_filter(image_data, image_name)
        elif noise_prediction == 'salt-pepper':
            # Apply Median filter. Padding is required for median filter.
            image_padded = pad(array=image_data, pad_width=1, mode='constant', constant_values=0)
            image_filtered = np.zeros(image_padded.shape, dtype=np.float32)
            SharedFunctions.median_filter(image_padded, 3, 1, image_filtered)
            image_filtered = image_filtered[1:-1, 1:-1]  # Remove padding now median filter done
        else:
            raise ValueError('Noise prediction does not match expected values')
        return image_filtered

    def train(self, train: List[DatasetManager.Image]):
        pass

    def classify(self, X) -> List[str]:
        pass
