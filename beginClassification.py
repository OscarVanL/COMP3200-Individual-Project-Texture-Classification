import getopt
import sys
import os
import numpy as np

import ClassificationUtils
from config import GlobalConfig
from data import DatasetManager
from algorithms import RLBP, MRELBP, BM3DELBP, NoiseClassifier
from algorithms.AlgorithmInterfaces import ImageProcessorInterface
from example import GenerateExamples
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
-r or --rotations : Whether to use rotated textures for the test images.
-n or --noise : Which type of noise to apply to test textures ('gaussian', 'speckle', 'salt-pepper')
-i or --intensity : How much noise to apply to test textures (Sigma / Variance / Ratio)
-m or --multiprocess : Whether to use multi process featurevector generation
-e or --example : Generate example images used in dissertation report
--debug : Whether to run in debug mode (uses a reduced dataset to speed up execution, prints more stuff)
"""


def main():
    # Parse Args.
    # 'scale' allows the image_scaled scale to be set. Eg: 0.25, 0.5, 1.0
    argList = sys.argv[1:]
    shortArg = 'a:d:t:s:S:rn:i:me'
    longArg = ['algorithm=', 'dataset=', 'train-ratio=', 'scale=', 'test-scale=', 'rotations', 'noise=', 'noise-intensity=', 'multiprocess', 'example',
               'debug']

    valid_algorithms = ['RLBP', 'MRLBP', 'MRELBP', 'BM3DELBP', 'NoiseClassifier']
    valid_datasets = ['kylberg']
    valid_noise = ['gaussian', 'speckle', 'salt-pepper']

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
                print('Using multiple processes for computing featurevectors')
                GlobalConfig.set('multiprocess', True)
            elif arg in ('-e', '--example'):
                print('Generating algorithm example image_scaled')
                GlobalConfig.set("examples", True)
            elif arg == '--debug':
                print("Running in debug mode")
                GlobalConfig.set('debug', True)
            else:
                raise ValueError('Unhandled argument provided:', arg)
    except getopt.error as err:
        print(str(err))

    if GlobalConfig.get('examples'):
        write_examples()

    # Load configured Dataset
    if GlobalConfig.get('dataset') == 'kylberg':
        if GlobalConfig.get('debug'):
            # To save time in debug mode, only load one class and load a smaller proportion of it (25% of samples)
            kylberg = DatasetManager.KylbergTextures(num_classes=2, data_ratio=0.25)
        else:
            kylberg = DatasetManager.KylbergTextures(num_classes=28, data_ratio=1.0)
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

    out_folder = os.path.join(os.getcwd(), 'out', GlobalConfig.get('algorithm'), dataset_folder)
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
    noise_out_dir = os.path.join(os.getcwd(), 'out', 'NoiseClassifier', dataset_folder,
                                 "scale-{}".format(int(GlobalConfig.get('scale') * 100)))
    test_noise_out_dir = os.path.join(os.getcwd(), 'out', 'NoiseClassifier', dataset_folder, algorithm.get_outdir(noisy_image, scaled_image))

    if GlobalConfig.get('multiprocess'):
        if GlobalConfig.get('algorithm') == 'NoiseClassifier' or GlobalConfig.get('algorithm') == 'BM3DELBP':
            with Pool(processes=4) as pool:
                # Generate image featurevectors and replace DatasetManager.Image with BM3DELBP.BM3DELBPImage
                dataset = pool.starmap(describe_noise, zip(dataset, repeat(noise_out_dir), repeat(test_noise_out_dir)))
        else:
            with Pool(processes=4) as pool:
                # Generate featurevectors
                dataset = pool.starmap(describe_image,
                                       zip(repeat(algorithm), dataset, repeat(train_out_dir), repeat(test_out_dir)))
    else:
        # Process the images without using multiprocessing Pools
        if GlobalConfig.get('algorithm') == 'NoiseClassifier' or GlobalConfig.get('algorithm') == 'BM3DELBP':
            for index, img in enumerate(dataset):
                # Generate image featurevectors and replace DatasetManager.Image with BM3DELBP.BM3DELBPImage
                dataset[index] = describe_noise(img, noise_out_dir, test_noise_out_dir)
        else:
            for index, img in enumerate(dataset):
                # Generate featurevetors
                describe_image(algorithm, img, train_out_dir, test_out_dir)


    # Train models and perform predictions
    if GlobalConfig.get('algorithm') == 'RLBP':
        predictor = RLBP.RobustLBPPredictor(dataset, cross_validator)
    elif GlobalConfig.get('algorithm') == 'MRLBP':
        print("Performing MRLBP Classification")
        predictor = RLBP.MultiresolutionLBPPredictor(dataset, cross_validator, 'svm')
    elif GlobalConfig.get('algorithm') == 'MRELBP':
        print("Performing MRELBP Classification")
        predictor = MRELBP.MedianRobustExtendedLBPPredictor(dataset, cross_validator)
    elif GlobalConfig.get('algorithm') == 'BM3DELBP':
        print("Performing BM3DELBP Classification")
        predictor = BM3DELBP.BM3DELBPPredictor(dataset, cross_validator)
    elif GlobalConfig.get('algorithm') == 'NoiseClassifier':
        print("Applying noise noise_classifier")
        predictor = BM3DELBP.NoiseTypePredictor(dataset, cross_validator)
    else:
        raise ValueError('Invalid algorithm choice')

    # Get the test label & test prediction for every fold of cross validation
    y_test, y_predicted = predictor.begin_cross_validation()

    if GlobalConfig.get('algorithm') == 'NoiseClassifier':
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
        os.path.join(os.getcwd(), 'data', 'kylberg', 'blanket1', 'blanket1-a-p001.png'))
    # Todo: Re-enable other example generation
    #ex.write_noise_examples()
    #ex.write_RLBP_example()
    #ex.write_MRLBP_example()
    #ex.write_MRELBP_example()
    ex.write_BM3DELBP_example()
    print("Finished generating examples")


def describe_image(algorithm: ImageProcessorInterface, image: DatasetManager.Image, train_out_dir: str, test_out_dir: str):
    """
    Applies an Algorithm to an image and writes the serialised featurevector to a directory.
    If a noise type is configured, a featurevector is computed for that noisy image too
    :param algorithm: Algorithm to apply
    :param image: Image
    :param train_out_dir: Directory to write serialised featurevectors of the image with no noise / scaling applied
    :param test_out_dir: Directory to read/write serialised featurevetors of the image with noise / scaling applied
    :return: Image after featurevectors generated/loaded
    """
    train_out_cat = os.path.join(train_out_dir, image.label)
    train_out_file = os.path.join(train_out_cat, '{}.npy'.format(image.name))
    test_out_cat = os.path.join(test_out_dir, image.label)
    test_out_file = os.path.join(test_out_cat, '{}.npy'.format(image.name))

    # Apply algorithm to normal image (or read from file if already applied before).
    if GlobalConfig.get('debug'):
        print("Read/Write train file to", train_out_file)
    try:
        image.featurevector = np.load(train_out_file)
        if GlobalConfig.get('debug'):
            print("Image featurevector loaded from file")

    except IOError:
        if GlobalConfig.get('debug'):
            print("Processing image", image.name)
        image.featurevector = algorithm.describe(image, test_image=False)

        # Make output folder if it doesn't exist
        if not (os.path.exists(train_out_cat)):
            os.makedirs(train_out_cat)

        np.save(train_out_file, image.featurevector)

    # If relevant, apply algorithm to test image (or read from file if already applied before).
    if (image.test_noise is not None) or (image.test_scale is not None):
        if GlobalConfig.get('debug'):
            print("Read/Write test file to", test_out_file)
        try:
            image.test_featurevector = np.load(test_out_file)
            if GlobalConfig.get('debug'):
                print("Noisy Image featurevector loaded from file")

        except IOError:
            if GlobalConfig.get('debug'):
                print("Processing image", image.name)
            image.test_featurevector = algorithm.describe(image, test_image=True)

            # Make output folder if it doesn't exist
            if not (os.path.exists(test_out_cat)):
                os.makedirs(test_out_cat)

            np.save(test_out_file, image.test_featurevector)

    # Ensure rotated variants of the image also have featurevectors generated/loaded
    if image.test_rotations is not None:
        for image in image.test_rotations:
            describe_image(algorithm, image, train_out_dir, test_out_dir)

    return image


def describe_noise(image: DatasetManager.Image, out_dir: str, test_out_dir: str) -> BM3DELBP.BM3DELBPImage:
    """
    Applies BM3DELBP's Noise Classifier to generate featurevectors for every image, for each type of noise applied.
    :param image: Image applied to
    :param out_dir: Directory to write serialised featurevectors
    :param test_out_dir: Directory to write serialised featurevector generated on test image
    :return: new_image : BM3DELBP.BM3DELBPImage for that image
    """
    new_image = BM3DELBP.BM3DELBPImage(image)
    noise_classifier = NoiseClassifier.NoiseClassifier()

    # Load / generate Gussian sigma 10 noise featurevector
    out_cat = os.path.join(out_dir, 'gaussian-10', image.label)
    out_noisy_image = os.path.join(out_cat, '{}-image.npy'.format(image.name))
    out_featurevector = os.path.join(out_cat, '{}-featurevector.npy'.format(image.name))
    if GlobalConfig.get('debug'):
        print("Read/Write to", out_noisy_image, "and", out_featurevector)
    try:
        new_image.gauss_10_noise_featurevector = np.load(out_featurevector)
        new_image.gauss_10_data = np.load(out_noisy_image)
        if GlobalConfig.get('debug'):
            print("Image featurevector loaded from file")
    except IOError:
        if GlobalConfig.get('debug'):
            print("Processing image", image.name)
        new_image.generate_gauss_10(noise_classifier)
        # Make output folder if it doesn't exist
        if not (os.path.exists(out_cat)):
            os.makedirs(out_cat)
        np.save(out_featurevector, new_image.gauss_10_noise_featurevector)
        np.save(out_noisy_image, new_image.gauss_10_data)

    # Load / generate Gaussian sigma 25 noise featurevector
    out_cat = os.path.join(out_dir, 'gaussian-25', image.label)
    out_noisy_image = os.path.join(out_cat, '{}-image.npy'.format(image.name))
    out_featurevector = os.path.join(out_cat, '{}-featurevector.npy'.format(image.name))
    if GlobalConfig.get('debug'):
        print("Read/Write to", out_noisy_image, "and", out_featurevector)
    try:
        new_image.gauss_25_noise_featurevector = np.load(out_featurevector)
        new_image.gauss_25_data = np.load(out_noisy_image)
        if GlobalConfig.get('debug'):
            print("Image featurevector loaded from file")
    except IOError:
        if GlobalConfig.get('debug'):
            print("Processing image", image.name)
        new_image.generate_gauss_25(noise_classifier)
        # Make output folder if it doesn't exist
        if not (os.path.exists(out_cat)):
            os.makedirs(out_cat)
        np.save(out_featurevector, new_image.gauss_25_noise_featurevector)
        np.save(out_noisy_image, new_image.gauss_25_data)

    # Load / generate Speckle 2% noise featurevector
    out_cat = os.path.join(out_dir, 'speckle-002', image.label)
    out_noisy_image = os.path.join(out_cat, '{}-image.npy'.format(image.name))
    out_featurevector = os.path.join(out_cat, '{}-featurevector.npy'.format(image.name))
    if GlobalConfig.get('debug'):
        print("Read/Write to", out_noisy_image, "and", out_featurevector)
    try:
        new_image.speckle_002_noise_featurevector = np.load(out_featurevector)
        new_image.speckle_002_data = np.load(out_noisy_image)
        if GlobalConfig.get('debug'):
            print("Image featurevector loaded from file")
    except IOError:
        if GlobalConfig.get('debug'):
            print("Processing image", image.name)
        new_image.generate_speckle_002(noise_classifier)
        # Make output folder if it doesn't exist
        if not (os.path.exists(out_cat)):
            os.makedirs(out_cat)
        np.save(out_featurevector, new_image.speckle_002_noise_featurevector)
        np.save(out_noisy_image, new_image.speckle_002_data)

    # Load / generate Salt and Pepper 2% noise featurevector
    out_cat = os.path.join(out_dir, 'salt-pepper-002', image.label)
    out_noisy_image = os.path.join(out_cat, '{}-image.npy'.format(image.name))
    out_featurevector = os.path.join(out_cat, '{}-featurevector.npy'.format(image.name))
    if GlobalConfig.get('debug'):
        print("Read/Write to", out_noisy_image, "and", out_featurevector)
    try:
        new_image.salt_pepper_002_noise_featurevector = np.load(out_featurevector)
        new_image.salt_pepper_002_data = np.load(out_noisy_image)
        if GlobalConfig.get('debug'):
            print("Image featurevector loaded from file")
    except IOError:
        if GlobalConfig.get('debug'):
            print("Processing image", image.name)
        new_image.generate_salt_pepper_002(noise_classifier)
        # Make output folder if it doesn't exist
        if not (os.path.exists(out_cat)):
            os.makedirs(out_cat)
        np.save(out_featurevector, new_image.salt_pepper_002_noise_featurevector)
        np.save(out_noisy_image, new_image.salt_pepper_002_data)

    # Load / generate featurevector on test image
    out_cat = os.path.join(test_out_dir, image.label)
    out_featurevector = os.path.join(out_cat, '{}-featurevector.npy'.format(image.name))
    if GlobalConfig.get('debug'):
        print("Read/Write to", out_noisy_image, "and", out_featurevector)
    try:
        new_image.test_noise_featurevector = np.load(out_featurevector)
        if GlobalConfig.get('debug'):
            print("Image featurevector loaded from file")
    except IOError:
        if GlobalConfig.get('debug'):
            print("Processing image", image.name)
        if new_image.test_data is None:
            raise ValueError('Image.test_data has not been assigned')
        new_image.generate_noise_featurevector(noise_classifier)
        # Make output folder if it doesn't exist
        if not (os.path.exists(out_cat)):
            os.makedirs(out_cat)
        np.save(out_featurevector, new_image.test_noise_featurevector)


    return new_image


if __name__ == '__main__':
    main()
