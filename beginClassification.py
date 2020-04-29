import getopt
import sys
import os
import psutil
import objgraph

from numba import config
config.THREADING_LAYER = 'workqueue'

import numpy as np

import ClassificationUtils
from config import GlobalConfig
from data import DatasetManager
from algorithms import RLBP, MRELBP, BM3DELBP, NoiseClassifier
from algorithms.AlgorithmInterfaces import ImageProcessorInterface
from example import GenerateExamples
from other import istarmap
import tqdm
from multiprocessing import Pool
from itertools import repeat


"""
This is the general launcher script for all IP functionality.

It parses the provided args, loads the relevant dataset, applies filters, applies the algorithm.

Args:
-a or --algorithm : Configure the algorithm to use.
-d or --dataset : Configure the Dataset to use
-t or --train-ratio : Ratio of the dataset to use for training. Eg: 0.8
-s or --scale : Amount to rescale the training textures. Eg: 0.5 to halve the resolution
-S or --test-scale : Amount to rescale the test textures. This is used for the scale invariance test
-k or --folds : Number of cross folds to complete
-r or --rotations : Whether to use rotated textures for the test images.
-n or --noise : Which type of noise to apply to test textures ('gaussian', 'speckle', 'salt-pepper')
-i or --noise-intensity : How much noise to apply to test textures (Sigma / Variance / Ratio)
-m or --multiprocess : Whether to use multi process featurevector generation
-e or --example : Generate example images used in dissertation report
--data-ratio : Ratio of the dataset to load.
--mrlbp-classifier : Which classifier to use for mrlbp, 'knn' or 'svm'
--noise-train : Apply the noise to the train dataset too
--ecs : If running on an ECS Lab machine, load the dataset from C:\Local instead of the CWD.
--debug : Whether to run in debug mode (uses a reduced dataset to speed up execution, prints more stuff)
"""

def main():
    # Parse Args.
    # 'scale' allows the image_scaled scale to be set. Eg: 0.25, 0.5, 1.0
    argList = sys.argv[1:]
    shortArg = 'a:d:t:s:S:k:rn:i:me'
    longArg = ['algorithm=', 'dataset=', 'train-ratio=', 'scale=', 'test-scale=', 'folds=', 'rotations', 'noise=',
               'noise-intensity=', 'multiprocess', 'example', 'data-ratio=', 'mrlbp-classifier=', 'noise-train', 'ecs', 'debug']

    valid_algorithms = ['RLBP', 'MRLBP', 'MRELBP', 'BM3DELBP', 'NoiseClassifier']
    valid_datasets = ['kylberg']
    valid_noise = ['gaussian', 'speckle', 'salt-pepper']
    valid_mrlbp_classifiers = ['svm', 'knn']

    try:
        args, vals = getopt.getopt(argList, shortArg, longArg)

        for arg, val in args:
            if arg in ('-a', '--algorithm'):
                if val in valid_algorithms:
                    print('Using algorithm:', val)
                    GlobalConfig.set("algorithm", val)
                else:
                    raise ValueError('Invalid algorithm configured, choose one of the following:', valid_algorithms)
            elif arg in ('-d', '--dataset'):
                if val in valid_datasets:
                    print("Using dataset:", val)
                    GlobalConfig.set("dataset", val)
                else:
                    raise ValueError('Invalid dataset configured, choose one of the following:', valid_datasets)
            elif arg in ('-t', '--train-test'):
                if 0 < float(val) <= 1.0:
                    print('Using train-ratio ratio of', val)
                    GlobalConfig.set('train_ratio', float(val))
                else:
                    raise ValueError('Train-test ratio must be 0 < train-test <= 1.0')
            elif arg in ('-s', '--scale'):
                if 0 < float(val) <= 1.0:
                    print('Using training image scale:', val)
                    GlobalConfig.set('scale', float(val))
                else:
                    raise ValueError('Scale must be 0 < scale <= 1.0')
            elif arg in ('-S', '--test-scale'):
                if 0 < float(val) <= 1.0:
                    print('Using testing image scale:', val)
                    GlobalConfig.set('test_scale', float(val))
                else:
                    raise ValueError('Test scale must be 0 < scale <= 1.0')
            elif arg in ('-k', '--folds'):
                print('Doing {} folds'.format(val))
                GlobalConfig.set("folds", int(val))
            elif arg in ('-r', '--rotations'):
                print('Using rotated image_scaled sources')
                GlobalConfig.set("rotate", True)
            elif arg in ('-n', '--noise'):
                if val in valid_noise:
                    print('Applying noise:', val)
                    GlobalConfig.set("noise", val)
                else:
                    raise ValueError('Invalid noise type, choose one of the following:', valid_noise)
            elif arg in ('-i', '--noise-intensity'):
                print('Using noise intensity (sigma / ratio) of:', val)
                GlobalConfig.set("noise_val", float(val))
            elif arg in ('-m', '--multiprocess'):
                cores = psutil.cpu_count()
                print('Using {} processor cores for computing featurevectors'.format(cores))
                GlobalConfig.set('multiprocess', True)
                GlobalConfig.set('cpu_count', cores)
            elif arg in ('-e', '--example'):
                print('Generating algorithm example image_scaled')
                GlobalConfig.set('examples', True)
            elif arg == '--data-ratio':
                if 0 < float(val) <= 1.0:
                    print('Using dataset ratio:', val)
                    GlobalConfig.set('data_ratio', float(val))
                else:
                    raise ValueError('Data ratio must be 0 < ratio <= 1.0')
            elif arg == '--mrlbp-classifier':
                if val in valid_mrlbp_classifiers:
                    print("MRLBP algorithm (if configured) will use {} classifier".format(val))
                    GlobalConfig.set('mrlbp_classifier', val)
                else:
                    raise ValueError('Invalid classifier chosen for mrlbp, choose one of the following:', valid_mrlbp_classifiers)
            elif arg == '--noise-train':
                print("Applying noise to the training dataset as well as the test dataset")
                GlobalConfig.set('train_noise', True)
            elif arg == '--ecs':
                print("Loading dataset from C:\Local")
                GlobalConfig.set('ECS', True)
            elif arg == '--debug':
                print("Running in debug mode")
                GlobalConfig.set('debug', True)
            else:
                raise ValueError('Unhandled argument provided:', arg)
    except getopt.error as err:
        print(str(err))

    if GlobalConfig.get('examples'):
        write_examples()

    if GlobalConfig.get('ECS'):
        GlobalConfig.set('CWD', r'\\filestore.soton.ac.uk\users\ojvl1g17\mydocuments\COMP3200-Texture-Classification')
    else:
        GlobalConfig.set('CWD', os.getcwd())

    # Load configured Dataset
    if GlobalConfig.get('dataset') == 'kylberg':
        if GlobalConfig.get('debug'):
            # To save time in debug mode, only load one class and load a smaller proportion of it (25% of samples)
            kylberg = DatasetManager.KylbergTextures(num_classes=2, data_ratio=0.25)
        else:
            kylberg = DatasetManager.KylbergTextures(num_classes=28, data_ratio=GlobalConfig.get('data_ratio'))
        # Load Dataset & Cross Validator
        dataset = kylberg.load_data()
        cross_validator = kylberg.get_cross_validator()

        print("Dataset loaded")
    elif GlobalConfig.get('dataset') is None:
        raise ValueError('No Dataset configured')
    else:
        raise ValueError('Invalid dataset')

    if GlobalConfig.get('rotate'):
        dataset_folder = GlobalConfig.get('dataset') + '-rotated'
    else:
        dataset_folder = GlobalConfig.get('dataset')

    out_folder = os.path.join(GlobalConfig.get('CWD'), 'out', GlobalConfig.get('algorithm'), dataset_folder)
    # Initialise algorithm
    if GlobalConfig.get('algorithm') == 'RLBP':
        print("Applying RLBP algorithm")
        algorithm = RLBP.RobustLBP()
    elif GlobalConfig.get('algorithm') == 'MRLBP':
        print("Applying MRLBP algorithm")
        algorithm = RLBP.MultiresolutionLBP(p=[8, 16, 24], r=[1, 2, 3])
    elif GlobalConfig.get('algorithm') == 'MRELBP':
        print("Applying MRELBP algorithm")
        algorithm = MRELBP.MedianRobustExtendedLBP(r1=[2, 4, 6, 8], p=8, w_center=3, w_r1=[3, 5, 7, 9])
    elif GlobalConfig.get('algorithm') == 'BM3DELBP':
        print("Applying BM3DELBP algorithm")
        algorithm = BM3DELBP.BM3DELBP()
    elif GlobalConfig.get('algorithm') == 'NoiseClassifier':
        # Noise Classifier is used in BM3DELBP algorithm usually, this allows for benchmarking of the classifier alone
        algorithm = NoiseClassifier.NoiseClassifier()
        pass
    else:
        raise ValueError('Invalid algorithm choice')

    # Get the Training out directory (i.e. Images without scaling/rotation/noise)
    train_out_dir = os.path.join(out_folder, algorithm.get_outdir(noisy_image=False, scaled_image=False))
    # Get the Testing out directory (i.e. Images with scaling/rotation/noise)
    if GlobalConfig.get('noise') is not None:
        noisy_image = True
    else:
        noisy_image = False
    if GlobalConfig.get('test_scale') is not None:
        scaled_image = True
    else:
        scaled_image = False
    test_out_dir = os.path.join(out_folder, algorithm.get_outdir(noisy_image, scaled_image))

    # Out path for noise classifier
    noise_out_dir = os.path.join(GlobalConfig.get('CWD'), 'out', 'NoiseClassifier', dataset_folder,
                                 "scale-{}".format(int(GlobalConfig.get('scale') * 100)))
    test_noise_out_dir = os.path.join(GlobalConfig.get('CWD'), 'out', 'NoiseClassifier', dataset_folder, algorithm.get_outdir(noisy_image, scaled_image))

    print("Replacing DatasetManager.Image with BM3DELBPImages")
    # Convert DatasetManager.Image into BM3DELBP.BM3DELBPImage
    if GlobalConfig.get('algorithm') == 'NoiseClassifier' or GlobalConfig.get('algorithm') == 'BM3DELBP':
        for index, img in enumerate(dataset):
            dataset[index] = BM3DELBP.BM3DELBPImage(img)
            # Also convert rotated images if necessary
            if img.test_rotations is not None:
                for index, rotated_img in enumerate(img.test_rotations):
                    img.test_rotations[index] = BM3DELBP.BM3DELBPImage(rotated_img)

    if GlobalConfig.get('multiprocess'):
        for index, img in enumerate(dataset):
            dataset[index] = (index, img)

        if GlobalConfig.get('algorithm') == 'NoiseClassifier' or GlobalConfig.get('algorithm') == 'BM3DELBP':
            with Pool(processes=GlobalConfig.get('cpu_count'), maxtasksperchild=20) as pool:
                # Generate image noise featurevectors
                for index, image in tqdm.tqdm(pool.istarmap(describe_noise_pool, zip(dataset, repeat(noise_out_dir), repeat(test_noise_out_dir))),
                                       total=len(dataset), desc='Noise Featurevectors'):
                    dataset[index] = image
        else:
            with Pool(processes=GlobalConfig.get('cpu_count'), maxtasksperchild=20) as pool:
                # Generate featurevectors
                for index, image in tqdm.tqdm(pool.istarmap(describe_image_pool, zip(repeat(algorithm), dataset, repeat(train_out_dir), repeat(test_out_dir))),
                                       total=len(dataset), desc='Texture Featurevectors'):
                    dataset[index] = image
    else:
        # Process the images without using multiprocessing Pools
        if GlobalConfig.get('algorithm') == 'NoiseClassifier' or GlobalConfig.get('algorithm') == 'BM3DELBP':
            for index, img in enumerate(dataset):
                # Generate image noise featurevectors
                describe_noise(img, noise_out_dir, test_noise_out_dir)
        else:
            for index, img in enumerate(dataset):
                # Generate featurevetors
                describe_image(algorithm, img, train_out_dir, test_out_dir)


    # Train models and perform predictions
    if GlobalConfig.get('algorithm') == 'RLBP':
        predictor = RLBP.RobustLBPPredictor(dataset, cross_validator)
    elif GlobalConfig.get('algorithm') == 'MRLBP':
        print("Performing MRLBP Classification")
        predictor = RLBP.MultiresolutionLBPPredictor(dataset, cross_validator)
    elif GlobalConfig.get('algorithm') == 'MRELBP':
        print("Performing MRELBP Classification")
        predictor = MRELBP.MedianRobustExtendedLBPPredictor(dataset, cross_validator)
    elif GlobalConfig.get('algorithm') == 'BM3DELBP':
        print("Performing BM3DELBP Classification")
        predictor = BM3DELBP.BM3DELBPPredictor(dataset, cross_validator)
    elif GlobalConfig.get('algorithm') == 'NoiseClassifier':
        print("Applying noise classifier")
        predictor = BM3DELBP.NoiseTypePredictor(dataset, cross_validator)
    else:
        raise ValueError('Invalid algorithm choice')

    # Get the test label & test prediction for every fold of cross validation
    y_test, y_predicted = predictor.begin_cross_validation()
    if GlobalConfig.get('algorithm') == 'NoiseClassifier':
        if GlobalConfig.get('noise') is None:
            classes = ['no-noise', 'gaussian', 'speckle', 'salt-pepper']
        else:
            classes = ['gaussian', 'speckle', 'salt-pepper']
    else:
        classes = kylberg.classes

    # Display confusion matrix
    ClassificationUtils.pretty_print_conf_matrix(y_test,
                                                 y_predicted,
                                                 classes,
                                                 title='{} Confusion Matrix'.format(GlobalConfig.get('algorithm')),
                                                 out_dir=test_out_dir)

    # Display classification report
    ClassificationUtils.make_classification_report(y_test, y_predicted, classes, test_out_dir)


def write_examples():
    """
    Generates example images for use in report
    :return: None
    """
    print("Generating algorithm example images")
    ex = GenerateExamples.GenerateExamples(
        os.path.join(GlobalConfig.get('CWD'), 'data', 'kylberg', 'blanket1', 'blanket1-a-p001.png'))
    # Todo: Re-enable other example generation
    #ex.write_noise_examples()
    #ex.write_RLBP_example()
    #ex.write_MRLBP_example()
    #ex.write_MRELBP_example()
    ex.write_BM3DELBP_example()
    print("Finished generating examples")


def describe_image_pool(algorithm, image_tuple, train_out_dir, test_out_dir):
    """
    A wrapper function for use in multiprocess Pool to maintain index IDs
    :param algorithm: See describe_image
    :param image_tuple: Tuple containing (index, DatasetManager.Image)
    :param train_out_dir: See describe_image
    :param test_out_dir: See describe_image
    :return: (index, DatasetManager.Image) after describing image
    """
    index, image = image_tuple
    image = describe_image(algorithm, image, train_out_dir, test_out_dir)
    return (index, image)


def describe_image(algorithm: ImageProcessorInterface, image: DatasetManager.Image, train_out_dir: str, test_out_dir: str):
    """
    Applies an Algorithm to an image and writes the serialised featurevector to a directory.
    If a noise type is configured, a featurevector is computed for that noisy image too
    :param algorithm: Algorithm to apply
    :param image: Image
    :param train_out_dir: Directory to write serialised featurevectors of the image with no noise / scaling applied
    :param test_out_dir: Directory to read/write serialised featurevetors of the image with noise / scaling applied
    :return: Image after attribute changes
    """
    train_out_cat = os.path.join(train_out_dir, image.label)
    train_out_file = os.path.join(train_out_cat, '{}.npy'.format(image.name))
    test_out_cat = os.path.join(test_out_dir, image.label)
    test_out_file = os.path.join(test_out_cat, '{}.npy'.format(image.name))

    # Apply algorithm to normal image (or read from file if already applied before).
    if GlobalConfig.get('debug'):
        print("Read/Write train file to", train_out_file)
    try:
        if GlobalConfig.get('train_noise'):
            image.featurevector = algorithm.describe(image, test_image=False)
            image.data = None
        else:
            image.featurevector = np.load(train_out_file, allow_pickle=True)
            image.data = None

            if GlobalConfig.get('debug'):
                print("Image featurevector loaded from file")

    except (IOError, ValueError):
        if GlobalConfig.get('debug'):
            print("Processing image", image.name)
        image.featurevector = algorithm.describe(image, test_image=False)
        image.data = None

        # Make output folder if it doesn't exist
        try:
            os.makedirs(train_out_cat)
        except FileExistsError:
            pass

        np.save(train_out_file, image.featurevector)

    # If relevant, apply algorithm to test image (or read from file if already applied before).
    if (image.test_noise is not None) or (image.test_scale is not None):
        if GlobalConfig.get('debug'):
            print("Read/Write test file to", test_out_file)
        try:
            image.test_featurevector = np.load(test_out_file, allow_pickle=True)
            # Remove test image data from memory, we don't need it anymore.
            image.test_data = None
            if GlobalConfig.get('debug'):
                print("Noisy Image featurevector loaded from file")

        except (IOError, ValueError):
            if GlobalConfig.get('debug'):
                print("Processing image", image.name)
            image.test_featurevector = algorithm.describe(image, test_image=True)
            # Remove test image data from memory, we don't need it anymore.
            image.test_data = None

            # Make output folder if it doesn't exist
            try:
                os.makedirs(test_out_cat)
            except FileExistsError:
                pass

            np.save(test_out_file, image.test_featurevector)

    # Ensure rotated variants of the image also have featurevectors generated/loaded
    if image.test_rotations is not None:
        for image in image.test_rotations:
            describe_image(algorithm, image, train_out_dir, test_out_dir)

    return image

def describe_noise_pool(image_tuple, out_dir, test_out_dir):
    """
    A wrapper function for use in multiprocess Pool to maintain index IDs
    :param image_tuple: Tuple containing (index, BM3DELBPImage)
    :param out_dir: See describe_noise()
    :param test_out_dir: See describe_noise()
    :return: (index, BM3DELBPImage) after describing noise
    """
    index, image = image_tuple
    image = describe_noise(image, out_dir, test_out_dir)
    return (index, image)

def describe_noise(image: BM3DELBP.BM3DELBPImage, out_dir: str, test_out_dir: str):
    """
    Applies BM3DELBP's Noise Classifier to generate featurevectors for every image, for each type of noise applied.
    :param image: BM3DELBPImage to apply noise to
    :param out_dir: Directory to write serialised featurevectors
    :param test_out_dir: Directory to write serialised featurevector generated on test image
    :return: BM3DELBPImage after attribute changes
    """
    noise_classifier = NoiseClassifier.NoiseClassifier()

    if GlobalConfig.get('noise') is None:
        # Generate non-noisy image noise featurevector
        out_cat = os.path.join(out_dir, 'no-noise', image.label)
        out_featurevector = os.path.join(out_cat, '{}-featurevector.npy'.format(image.name))
        if GlobalConfig.get('debug'):
            print("Read/Write to", out_featurevector)
        try:
            # Try loading serialised featurevectors if it's ran before already
            image.no_noise_featurevector = np.load(out_featurevector, allow_pickle=True)
            if GlobalConfig.get('debug'):
                print("Image featurevector loaded from file")
        except (IOError, ValueError):
            if GlobalConfig.get('debug'):
                print("Processing iamge", image.name)
            image.generate_normal_featurevector(noise_classifier)
            # Make output folder if it doesn't exist
            try:
                os.makedirs(out_cat)
            except FileExistsError:
                pass
            np.save(out_featurevector, image.no_noise_featurevector)

    # Use a smaller NoiseClassifier training dataset if testing rotated images
    if not GlobalConfig.get('rotate'):
        # Load / generate Gussian sigma 10 noise featurevector
        out_cat = os.path.join(out_dir, 'gaussian-10', image.label)
        out_featurevector = os.path.join(out_cat, '{}-featurevector.npy'.format(image.name))
        if GlobalConfig.get('debug'):
            print("Read/Write to", out_featurevector)
        try:
            image.gauss_10_noise_featurevector = np.load(out_featurevector, allow_pickle=True)
            if GlobalConfig.get('debug'):
                print("Image featurevector loaded from file")
        except (IOError, ValueError):
            if GlobalConfig.get('debug'):
                print("Processing image", image.name)
            image.generate_gauss_10(noise_classifier)
            try:
                os.makedirs(out_cat)
            except FileExistsError:
                pass
            np.save(out_featurevector, image.gauss_10_noise_featurevector)

        # Load / generate Speckle var=0.02 noise featurevector
        out_cat = os.path.join(out_dir, 'speckle-002', image.label)
        out_featurevector = os.path.join(out_cat, '{}-featurevector.npy'.format(image.name))
        if GlobalConfig.get('debug'):
            print("Read/Write to", out_featurevector)
        try:
            image.speckle_002_noise_featurevector = np.load(out_featurevector, allow_pickle=True)
            if GlobalConfig.get('debug'):
                print("Image featurevector loaded from file")
        except (IOError, ValueError):
            if GlobalConfig.get('debug'):
                print("Processing image", image.name)
            image.generate_speckle_002(noise_classifier)
            try:
                os.makedirs(out_cat)
            except FileExistsError:
                pass
            np.save(out_featurevector, image.speckle_002_noise_featurevector)

        # Load / generate Salt and Pepper 2% noise featurevector
        out_cat = os.path.join(out_dir, 'salt-pepper-002', image.label)
        out_featurevector = os.path.join(out_cat, '{}-featurevector.npy'.format(image.name))
        if GlobalConfig.get('debug'):
            print("Read/Write to", out_featurevector)
        try:
            image.salt_pepper_002_noise_featurevector = np.load(out_featurevector, allow_pickle=True)
            if GlobalConfig.get('debug'):
                print("Image featurevector loaded from file")
        except (IOError, ValueError):
            if GlobalConfig.get('debug'):
                print("Processing image", image.name)
            image.generate_salt_pepper_002(noise_classifier)
            try:
                os.makedirs(out_cat)
            except FileExistsError:
                pass
            np.save(out_featurevector, image.salt_pepper_002_noise_featurevector)

    # Load / generate Gaussian sigma 25 noise featurevector
    out_cat = os.path.join(out_dir, 'gaussian-25', image.label)
    out_featurevector = os.path.join(out_cat, '{}-featurevector.npy'.format(image.name))
    if GlobalConfig.get('debug'):
        print("Read/Write to", out_featurevector)
    try:
        image.gauss_25_noise_featurevector = np.load(out_featurevector, allow_pickle=True)
        if GlobalConfig.get('debug'):
            print("Image featurevector loaded from file")
    except (IOError, ValueError):
        if GlobalConfig.get('debug'):
            print("Processing image", image.name)
        image.generate_gauss_25(noise_classifier)
        try:
            os.makedirs(out_cat)
        except FileExistsError:
            pass
        np.save(out_featurevector, image.gauss_25_noise_featurevector)

    # Load / generate Speckle var=0.04 noise featurevector
    out_cat = os.path.join(out_dir, 'speckle-004', image.label)
    out_featurevector = os.path.join(out_cat, '{}-featurevector.npy'.format(image.name))
    if GlobalConfig.get('debug'):
        print("Read/Write to", out_featurevector)
    try:
        image.speckle_004_noise_featurevector = np.load(out_featurevector, allow_pickle=True)
        if GlobalConfig.get('debug'):
            print("Image featurevector loaded from file")
    except (IOError, ValueError):
        if GlobalConfig.get('debug'):
            print("Processing image", image.name)
        image.generate_speckle_004(noise_classifier)
        try:
            os.makedirs(out_cat)
        except FileExistsError:
            pass
        np.save(out_featurevector, image.speckle_004_noise_featurevector)

    # Load / generate Salt and Pepper 4% noise featurevector
    out_cat = os.path.join(out_dir, 'salt-pepper-004', image.label)
    out_featurevector = os.path.join(out_cat, '{}-featurevector.npy'.format(image.name))
    if GlobalConfig.get('debug'):
        print("Read/Write to", out_featurevector)
    try:
        image.salt_pepper_004_noise_featurevector = np.load(out_featurevector, allow_pickle=True)
        if GlobalConfig.get('debug'):
            print("Image featurevector loaded from file")
    except (IOError, ValueError):
        if GlobalConfig.get('debug'):
            print("Processing image", image.name)
        image.generate_salt_pepper_004(noise_classifier)
        try:
            os.makedirs(out_cat)
        except FileExistsError:
            pass
        np.save(out_featurevector, image.salt_pepper_004_noise_featurevector)

    # Load / generate featurevector on test image
    out_cat = os.path.join(test_out_dir, image.label)
    out_featurevector = os.path.join(out_cat, '{}-featurevector.npy'.format(image.name))
    if GlobalConfig.get('debug'):
        print("Read/Write to", out_featurevector)
    try:
        image.test_noise_featurevector = np.load(out_featurevector, allow_pickle=True)
        if GlobalConfig.get('debug'):
            print("Image featurevector loaded from file")
    except (IOError, ValueError):
        if GlobalConfig.get('debug'):
            print("Processing image", image.name)
        if image.test_data is None:
            raise ValueError('Image.test_data has not been assigned')
        image.generate_noise_featurevector(noise_classifier)
        try:
            os.makedirs(out_cat)
        except FileExistsError:
            pass
        np.save(out_featurevector, image.test_noise_featurevector)

    # Ensure rotated variants of the image also have noise featurevectors generated/loaded
    if image.test_rotations is not None:
        for image in image.test_rotations:
            describe_noise(image, out_dir, test_out_dir)

    return image


if __name__ == '__main__':
    main()
