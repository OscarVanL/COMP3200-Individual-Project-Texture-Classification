import os
from statistics import mean
import random
import cv2
import glob
from config import GlobalConfig
import data.ImageUtils as ImageUtils
from sklearn.model_selection import StratifiedShuffleSplit


class Image:
    def __init__(self, data, name, label):
        """
        Image and its metadata
        :param data: ndarray of original image data
        :param name: Name of image (eg: blanket1-a-p001)
        :param label: Category/label of image (eg: blanket1)
        """
        self.data = data
        self.name = name
        self.label = label
        self.featurevector = None  # Store featurevector calculated on normal image
        self.test_data = None  # Store test image with applied alterations (eg: Scale, noise)
        self.test_featurevector = None  # Store featurevector calculated on test image
        self.test_noise = None
        self.test_noise_val = None
        self.test_rotations = None
        self.test_scale = None


class KylbergTextures:
    """
    num_classes: Number of classes to use, fewer classes makes it easier. Kylberg has 28 classes.
    train_test_ratio: Ratio of data to use for training. Eg: 0.8 = 80% train
    data_ratio: Proportion of the dataset to use (eg: 0.5 = 50% of the dataset used), can help reduce training time
    resize_ratio: Amount to resize image_scaled by. Eg: 0.25, 0.5, or 1.0 (full-size).

    """

    def __init__(self, num_classes: int, data_ratio: float):
        self.num_categories = num_classes
        self.data_ratio = data_ratio
        self.classes = []
        self.train = []
        self.test = []

        if GlobalConfig.get('rotate'):
            print("Using rotated textures")
            if GlobalConfig.get('ECS'):
                self.KYLBERG_DIR = os.path.join('C:/', 'Local', 'data', 'kylberg-rotated')
            else:
                self.KYLBERG_DIR = os.path.join(GlobalConfig.get('CWD'), 'data', 'kylberg-rotated')
        else:
            print("Using non-rotated textures")
            if GlobalConfig.get('ECS'):
                self.KYLBERG_DIR = os.path.join('C:/', 'Local', 'data', 'kylberg')
            else:
                self.KYLBERG_DIR = os.path.join(GlobalConfig.get('CWD'), 'data', 'kylberg')
        print('Using {} scale'.format(GlobalConfig.get('scale')))
        if GlobalConfig.get('noise') is not None:
            print('Applying {} noise'.format(GlobalConfig.get('noise')))

    def load_data(self):
        """
        Loads each image in the dataset
        :return: List of loaded images, packed into DatasetManager.Image objects.
        """
        # Find classes in the dataset
        categories = os.listdir(self.KYLBERG_DIR)
        avg_cat_size = mean([len(glob.glob(os.path.join(self.KYLBERG_DIR, cat, "") + "*.png")) for cat in categories])
        print('{} categories in Kylberg, with average {} images per label.'.format(len(categories), avg_cat_size))
        if self.num_categories < len(categories):
            # Select num_classes randomly if opting to use a smaller proportion of the dataset's classes.
            self.classes = random.sample(categories, self.num_categories)
        else:
            self.classes = categories

        print('Loading {} categories:'.format(self.num_categories), self.classes)

        # Enumerate dataset containing loaded images for each label
        images = []
        for cat in self.classes:
            # If rotation is enabled, load images with the same prefix name as one DatasetManager.Image
            if GlobalConfig.get('rotate'):
                # Get paths for all images ending with r000.png (unrotated)
                unrotated_paths = glob.glob(os.path.join(self.KYLBERG_DIR, cat, "") + "*r000.png")
                # Load the unrotated image, then find all 12 rotated variants and set these as a parameter
                for unrotated_img_path in unrotated_paths:
                    main_img = self.load_image(unrotated_img_path)
                    img_variants = glob.glob(unrotated_img_path[:-7] + "*.png")
                    if len(img_variants) == 0:
                        raise ValueError('Could not find rotated variants of the image')
                    img_variants = [self.load_image(img) for img in img_variants]
                    main_img.test_rotations = img_variants
                    images.append(main_img)
            else:
                # Get paths of all images in this label
                img_paths = glob.glob(os.path.join(self.KYLBERG_DIR, cat, "") + "*.png")
                for path in img_paths:
                    images.append(self.load_image(path))

        if len(images) == 0:
            raise FileNotFoundError('No images were loaded from the dataset. Is it in the correct location?')
        return images

    @staticmethod
    def get_cross_validator():
        """
        Generates and returns the ShuffleSplit cross validator.
        This uses a fixed random_state=0, so that every time each cross fold contains the same subset of data.
        :return: sklearn.model_selection.ShuffleSplit with relevant parameters.
        """
        return StratifiedShuffleSplit(n_splits=GlobalConfig.get('folds'), train_size=GlobalConfig.get('train_ratio'), random_state=0)

    @staticmethod
    def load_image(path):
        """
        Load a single greyscale image_scaled with the defined resize ratio, scales image_scaled to zero mean and unit variance
        and adds noise type (if configured).
        :param path: Path to the image
        :return: Loaded image_scaled, np.float32 ndarray.
        """
        image_name = path.split(os.sep)[-1].partition('.')[0]
        image_label = path.split(os.sep)[-2]  # Gets the image category
        # Scale train image
        train_data = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (0, 0),
                                fx=GlobalConfig.get('scale'),
                                fy=GlobalConfig.get('scale'))
        # Scale test image
        if GlobalConfig.get('test_scale') is not None:
            test_data = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (0, 0),
                                   fx=GlobalConfig.get('test_scale'),
                                   fy=GlobalConfig.get('test_scale'))
        else:
            test_data = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (0, 0),
                                   fx=GlobalConfig.get('scale'),
                                   fy=GlobalConfig.get('scale'))

        algo = GlobalConfig.get('algorithm')
        # Convert image from uint8 to float32
        if algo == 'MRELBP' or algo == 'BM3DELBP' or algo == 'NoiseClassifier':
            #  Image gets scaled to 0 mean if using MRELBP, BM3DELBP or BM3DELBP's NoiseClassifier
            train_data = ImageUtils.scale_uint8_image_float32(train_data)
            test_data = ImageUtils.scale_uint8_image_float32(test_data)
        else:
            # Other algorithms do not require this scaling.
            train_data = ImageUtils.convert_uint8_image_float32(train_data)
            test_data = ImageUtils.convert_uint8_image_float32(test_data)

        # Apply filters
        if GlobalConfig.get('noise') is None:
            pass
        elif GlobalConfig.get('noise') == 'gaussian':
            test_data = ImageUtils.add_gaussian_noise_skimage(test_data, GlobalConfig.get('noise_val'))
        elif GlobalConfig.get('noise') == 'speckle':
            test_data = ImageUtils.add_speckle_noise_skimage(test_data, GlobalConfig.get('noise_val'))
        elif GlobalConfig.get('noise') == 'salt-pepper':
            test_data = ImageUtils.add_salt_pepper_noise_skimage(test_data, GlobalConfig.get('noise_val'))
        else:
            raise ValueError('Invalid image_scaled noise type defined')

        loaded_image = Image(train_data, image_name, image_label)
        loaded_image.test_data = test_data
        loaded_image.test_scale = GlobalConfig.get('test_scale')
        loaded_image.test_noise = GlobalConfig.get('noise')
        loaded_image.test_noise_val = GlobalConfig.get('noise_val')
        return loaded_image
