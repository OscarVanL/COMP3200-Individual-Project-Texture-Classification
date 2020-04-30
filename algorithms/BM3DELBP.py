from typing import List, Tuple

from algorithms import SharedFunctions, SARBM3D, MRELBP
from algorithms.AlgorithmInterfaces import ImageProcessorInterface, ImageClassifierInterface
from algorithms.NoiseClassifier import NoiseTypePredictor
from config import GlobalConfig
from sklearn.svm import SVC
from skimage.util import pad
from other import istarmap
import tqdm
from multiprocessing import Pool
from itertools import repeat
import numpy as np
from data import ImageUtils, DatasetManager
import os.path


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

        if GlobalConfig.get('train_noise'):
            return "scale-{}_noise-{}_noiseval-{}-trainnoise".format(image_scale, noise_type, noise_val)
        else:
            return "scale-{}_noise-{}_noiseval-{}".format(image_scale, noise_type, noise_val)

    def describe_filter(self, image, test_image: bool, train_out_dir, test_out_dir, ecs=False):
        """
        Perform the filtering + description stage for BM3DELBP algorithm.
        Load/Save featurevector to file to save time on future runs
        :param image: Image data to apply BM3DELBP filter + description upon
        :param test_image: Whether the featurevector is being generated for the test image
        :param dataset: Name of dataset folder
        :param ecs: Whether this is running on ECS machines, ignore this.
        :return: Image containing featurevector
        """
        if image is None:
            raise ValueError("Image passed into describe_filter is NoneType")

        if not test_image and image.featurevector is None:
            # Non-noisy original image for training
            out_file = os.path.join(train_out_dir, '{}.npy'.format(image.name))
            # Read/generate featurevector for image
            if GlobalConfig.get('debug'):
                print("Read/Write BM3DELBP featurevector file to", out_file)
            try:
                image.featurevector = np.load(out_file, allow_pickle=True)
                image.data = None  # Unassign image data as it's no longer needed
                if GlobalConfig.get('debug'):
                    print("Image featurevector loaded from file")
            except (IOError, ValueError):
                if GlobalConfig.get('debug'):
                    print("Processing image", image.name)
                image.featurevector = self.describe(image.data, test_image=False)
                image.data = None  # Unassign image data as it's no longer needed

                # Make output folder if it doesn't exist
                try:
                    os.makedirs(train_out_dir)
                except FileExistsError:
                    pass

                np.save(out_file, image.featurevector)


        # Check test_featurevector isn't already assigned, as it may have already been assigned in a previous fold
        if test_image and image.test_featurevector is None:
            # When storing test image generated featurevectors, store the featurevectors according to their noise type
            # prediction.
            test_out_dir = os.path.join(test_out_dir, self.get_filter_name(image.noise_prediction))
            out_file = os.path.join(test_out_dir, '{}.npy'.format(image.name))

            # Read/generate featurevector for test image
            if GlobalConfig.get('debug'):
                print("Read/Write BM3DELBP featurevector file to", out_file)
            try:
                image.test_featurevector = np.load(out_file, allow_pickle=True)
                image.test_data = None
                if GlobalConfig.get('debug'):
                    print("Image featurevector loaded from file")
            except (IOError, ValueError):
                if GlobalConfig.get('debug'):
                    print("Processing image", image.name)
                # Perform appropriate filter
                if image.noise_prediction is None:
                    raise ValueError('describe_filter called on BM3DELBP image where no noise prediction has been made')
                test_data_filtered = self.apply_filter(image.test_data, image.name, image.noise_prediction, ecs)
                image.test_featurevector = self.describe(test_data_filtered, test_image=True)
                image.data = None

                # Make output folder if it doesn't exist
                try:
                    os.makedirs(test_out_dir)
                except FileExistsError:
                    pass

                np.save(out_file, image.test_featurevector)

        # Also process rotated images, but only for testing (since we're testing for rotation invariance)
        if image.test_rotations is not None:
            if test_image:
                for image in image.test_rotations:
                    self.describe_filter(image, test_image, train_out_dir, test_out_dir)

        return image

    def describe(self, image, test_image: bool):
        """
        Essentially BM3DELBP's descriptors are the same as that in MRELBP!
        The primary difference is the noise identification + filtering stage.
        :param image: Image's Numpy ndarray data to classify
        :param test_image: Whether this is a test image with a noise type applied (Not currently in use)
        :return:
        """

        if self.save_img:
            # Todo: Implement example image generation for report
            pass

        # Zero-pad image_scaled with padding border. Required for RELBP descriptor
        image_padded = pad(array=image, pad_width=self.elbp.padding, mode='constant', constant_values=0)
        image_padded = image_padded.astype(np.float32)
        return self.elbp.calculate_relbp(image_padded)

    def get_filter_name(self, noise_prediction):
        if noise_prediction == 'gaussian':
            return 'BM3D'
        elif noise_prediction == 'speckle':
            return 'SARBM3D'
        elif noise_prediction == 'salt-pepper':
            return 'median'
        elif noise_prediction == 'no-noise':
            return 'no-filter'
        else:
            raise ValueError('Noise prediction {} does not match expected values'.format(noise_prediction))

    def apply_filter(self, image_data, image_name, noise_prediction, ecs=False):
        if noise_prediction == 'gaussian':
            # Apply BM3D filter
            image_filtered = SharedFunctions.bm3d_filter(image_data, 50/255)
        elif noise_prediction == 'speckle':
            # Establish connection to MATLAB Engine for SAR-BM3D filter
            sar_bm3d = SARBM3D.SARBM3DFilter(ecs)
            # Connect to MATLAB Engine for SAR-BM3D filter
            sar_bm3d.connect_matlab()
            # Apply SAR-BM3D filter
            image_filtered = sar_bm3d.sar_bm3d_filter(image_data, image_name)
            try:
                # Disconnect from MATLAB Engine
                sar_bm3d.disconnect_matlab()
            except SystemError:
                # If we try to disconnect matlab after it has already disconnected (eg: crashed) this throws an exception
                pass
        elif noise_prediction == 'salt-pepper':
            # Apply Median filter. Padding is required for median filter.
            image_padded = pad(array=image_data, pad_width=1, mode='constant', constant_values=0)
            image_filtered = np.zeros(image_padded.shape, dtype=np.float32)
            SharedFunctions.median_filter(image_padded, 3, 1, image_filtered)
            image_filtered = image_filtered[1:-1, 1:-1]  # Remove padding now median filter done
        elif noise_prediction == 'no-noise':
            image_filtered = image_data.copy()
        else:
            raise ValueError('Noise prediction does not match expected values')
        return image_filtered


class BM3DELBPImage(DatasetManager.Image):
    def __init__(self, image: DatasetManager.Image):
        """
        A modified version of DatasetManger.Image class specific to the BM3DELBP implementation.
        This takes an Image, applies each noise type, and for each image generates the featurevector for the NoiseClassifier.
        Doing this for each fold would be enormously slow, so it allows this to be reused.
        """
        super().__init__(image.data, image.name, image.label)
        # Import attributes from DatasetManager.Image
        self.featurevector = image.featurevector
        self.test_data = image.test_data
        self.test_featurevector = image.test_featurevector
        self.test_noise = image.test_noise
        self.test_noise_val = image.test_noise_val
        self.test_rotations = image.test_rotations
        self.test_scale = image.test_scale

        self.test_noise_featurevector = None  # Featurevector for the noise identifier generated on test image
        self.noise_prediction = None  # Prediction Noise Classifier made for original image data

        self.no_noise_featurevector = None
        self.no_noise_prediction = None
        self.gauss_10_noise_featurevector = None  # Featurevector for the noise identifier
        self.gauss_10_prediction = None  # When the Noise Classifier has predicted the noise type, store it here
        self.gauss_25_noise_featurevector = None
        self.gauss_25_prediction = None
        self.speckle_002_noise_featurevector = None
        self.speckle_002_prediction = None
        self.speckle_004_noise_featurevector = None
        self.speckle_004_prediction = None
        self.salt_pepper_002_noise_featurevector = None
        self.salt_pepper_002_prediction = None
        self.salt_pepper_004_noise_featurevector = None
        self.salt_pepper_004_prediction = None

    # Generate the noise featurevector for the test image to predict with the classifier
    def generate_noise_featurevector(self, noise_classifier):
        self.test_noise_featurevector = noise_classifier.describe(self.test_data, test_image=True)

    # Generate noise featurevectors on image with no noise applied to detect non-noisy images
    def generate_normal_featurevector(self, noise_clasifier):
        self.no_noise_featurevector = noise_clasifier.describe(self.data, test_image=False)

    # Generate noise on the non-test image (no transformations) to train the classifier with
    def generate_gauss_10(self, noise_classifier):
        gauss_10_data = ImageUtils.add_gaussian_noise_skimage(self.data, 10)
        self.gauss_10_noise_featurevector = noise_classifier.describe(gauss_10_data, test_image=False)

    def generate_gauss_25(self, noise_classifier):
        gauss_25_data = ImageUtils.add_gaussian_noise_skimage(self.data, 25)
        self.gauss_25_noise_featurevector = noise_classifier.describe(gauss_25_data, test_image=False)

    def generate_speckle_002(self, noise_classifier):
        speckle_data = ImageUtils.add_speckle_noise_skimage(self.data, 0.02)
        self.speckle_002_noise_featurevector = noise_classifier.describe(speckle_data, test_image=False)

    def generate_speckle_004(self, noise_classifier):
        speckle_data = ImageUtils.add_speckle_noise_skimage(self.data, 0.04)
        self.speckle_004_noise_featurevector = noise_classifier.describe(speckle_data, test_image=False)

    def generate_salt_pepper_002(self, noise_classifier):
        salt_pepper_002_data = ImageUtils.add_salt_pepper_noise_skimage(self.data, 0.02)
        self.salt_pepper_002_noise_featurevector = noise_classifier.describe(salt_pepper_002_data, test_image=False)

    def generate_salt_pepper_004(self, noise_classifier):
        salt_pepper_004_data = ImageUtils.add_salt_pepper_noise_skimage(self.data, 0.04)
        self.salt_pepper_004_noise_featurevector = noise_classifier.describe(salt_pepper_004_data, test_image=False)


class BM3DELBPPredictor(ImageClassifierInterface):

    def __init__(self, dataset: List[BM3DELBPImage], cross_validator):
        super().__init__(dataset, cross_validator)
        self.BM3DELBP = BM3DELBP()
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

        if GlobalConfig.get('rotate'):
            dataset = GlobalConfig.get('dataset') + '-rotated'
        else:
            dataset = GlobalConfig.get('dataset')

        train_out_dir = os.path.join(GlobalConfig.get('CWD'), 'out', 'BM3DELBP', dataset,
                               self.BM3DELBP.get_outdir(noisy_image=False, scaled_image=False))

        noisy_image = GlobalConfig.get('noise') is not None
        scaled_image = GlobalConfig.get('test_scale') is not None
        test_out_dir = os.path.join(GlobalConfig.get('CWD'), 'out', 'BM3DELBP', dataset,
                               self.BM3DELBP.get_outdir(noisy_image=noisy_image, scaled_image=scaled_image))

        fold = 1
        test_X_all = []
        test_y_all = []
        pred_y_all = []

        for train_index, test_index in self.cross_validator.split(self.dataset, self.dataset_y):
            print("Performing fold", fold)
            # Train NoiseClassifier on Train indexes only
            self.noise_classifier = None
            noise_classifier_train = [self.dataset[index] for index in train_index]
            self.noise_classifier = NoiseTypePredictor(noise_classifier_train, None)
            self.noise_classifier.train()

            print("Noise Classifier trained")
            # Remove any existing BM3DELBP classifier
            self.classifier = None
            # Train on this fold
            train = []
            if GlobalConfig.get('multiprocess'):
                with Pool(GlobalConfig.get('cpu_count'), maxtasksperchild=5) as pool_train:
                    # Generate featurevectors
                    for image in tqdm.tqdm(pool_train.istarmap(self.BM3DELBP.describe_filter,
                                                         zip([self.dataset[index] for index in train_index], repeat(False), repeat(train_out_dir), repeat(test_out_dir), repeat(GlobalConfig.get('ECS')))),
                                           total=len(train_index), desc='BM3DELBP Train Featurevectors'):
                        train.append(image)
            else:
                for index in train_index:
                    train.append(self.BM3DELBP.describe_filter(image=self.dataset[index], test_image=False, train_out_dir=train_out_dir, test_out_dir=test_out_dir, ecs=GlobalConfig.get('ECS')))
            self.train(train)

            print("BM3DELBP Classifier trained on non-noisy images")
            print("Making noise predictions for test images")

            test_X = []
            test_y = []

            # Make noise prediction for each of the test images
            for index in test_index:
                image = self.dataset[index]
                # Reshaping is required because we're classifying a single item at a time
                image.noise_prediction = self.noise_classifier.classify(image.test_noise_featurevector.reshape(1, -1))

            print("Noise predictions made on test images")
            print("Applying relevant filter and generating BM3DELBP descriptor for test images")

            # Apply BM3DELBP filter & generate BM3DELBP descriptor
            if GlobalConfig.get('multiprocess'):
                # Generate test_featurevectors using multiprocessing
                with Pool(GlobalConfig.get('cpu_count'), maxtasksperchild=5) as pool_test:
                    for image in tqdm.tqdm(pool_test.istarmap(self.BM3DELBP.describe_filter,
                                                         zip([self.dataset[index] for index in test_index], repeat(True), repeat(train_out_dir), repeat(test_out_dir), repeat(GlobalConfig.get('ECS')))),
                                            total=len(test_index), desc='BM3DELBP Test Featurevectors'):
                        test_X.append(image.test_featurevector)
                        test_y.append(image.label)
                        # Also add rotations of the image if they exist
                        if image.test_rotations is not None:
                            for rotated_image in image.test_rotations:
                                print("ROTATED IMAGES LENGTH (should be 12): ", len(rotated_image))
                                test_X.append(rotated_image.test_featurevector)
                                test_y.append(image.label)
            else:
                for index in test_index:
                    # Apply BM3DELBP's appropriate filter and generate the BM3DELBP descriptor
                    self.dataset[index] = self.BM3DELBP.describe_filter(image=self.dataset[index], test_image=True, train_out_dir=train_out_dir, test_out_dir=test_out_dir, ecs=GlobalConfig.get('ECS'))
                    test_X.append(self.dataset[index].test_featurevector)
                    test_y.append(self.dataset[index].label)
                    # Also add rotations of the image if they exist
                    if self.dataset[index].test_rotations is not None:
                        for rotated_image in self.dataset[index].test_rotations:
                            test_X.append(rotated_image.test_featurevector)
                            test_y.append(image.label)

            pred_y = self.classify(test_X)
            test_X_all.extend(test_X)
            test_y_all.extend(test_y)
            pred_y_all.extend(pred_y)

            print("Test values classified")

            fold += 1

        return test_y_all, pred_y_all

    def train(self, train: List[DatasetManager.Image]):
        X_train = [img.featurevector for img in train]
        # Convert List[narray] to ndarray
        X_train = np.stack(X_train, axis=0)
        X_train = X_train.astype(np.float64)
        y_train = [img.label for img in train]
        self.classifier = SVC(kernel='poly')
        self.classifier.fit(X_train, y_train)

    def classify(self, X) -> List[str]:
        return self.classifier.predict(X)
