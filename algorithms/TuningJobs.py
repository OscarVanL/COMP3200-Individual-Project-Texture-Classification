import os
import gc

from numba import config
config.THREADING_LAYER = 'workqueue'

from algorithms.NoiseClassifier import NoiseTypePredictor, NoiseClassifier
from algorithms import BM3DELBP
from config import GlobalConfig
from data import DatasetManager, ImageUtils
from other import istarmap
import tqdm
from multiprocessing import Pool
from itertools import repeat
from sklearn.metrics import f1_score
import psutil
import csv

def do_classification(settings_tuple, noise_classifier, cross_validator, images):
    sigma, cutoff, a, b = settings_tuple
    sigma_val = sigma / 255
    X_dataset = []
    y_dataset = []

    for image in images:
        X_dataset.append(noise_classifier.describe(ImageUtils.add_gaussian_noise_skimage(image.data, 10), sigma_val, cutoff, a, b))
        y_dataset.append('gaussian')
        X_dataset.append(noise_classifier.describe(ImageUtils.add_speckle_noise_skimage(image.data, 0.02), sigma_val, cutoff, a, b))
        y_dataset.append('speckle')
        X_dataset.append(noise_classifier.describe(ImageUtils.add_salt_pepper_noise_skimage(image.data, 0.02), sigma_val, cutoff, a, b))
        y_dataset.append('salt-pepper')

    print("Training with Sigma: {}/255, Cutoff Frequency: {}, a: {}, b: {}".format(sigma, cutoff, a, b))
    noise_classifier = NoiseTypePredictor(X_dataset, y_dataset, cross_validator)
    test_y_all, pred_y_all = noise_classifier.begin_cross_validation()
    f1 = f1_score(test_y_all, pred_y_all, labels=['gaussian', 'speckle', 'salt-pepper'])
    print("Completed with F1: {} , Sigma: {}, Cutoff freq: {}, a: {}, b: {}")
    return (f1, sigma, cutoff, a, b)

def tune_noise_classifier():
    GlobalConfig.set('dataset', 'kylberg')
    GlobalConfig.set('ECS', True)
    GlobalConfig.set('algorithm', 'NoiseClassifier')
    GlobalConfig.set('scale', 0.5)
    GlobalConfig.set('CWD', r'\\filestore.soton.ac.uk\users\ojvl1g17\mydocuments\COMP3200-Texture-Classification')
    #GlobalConfig.set('CWD', os.getcwd())
    GlobalConfig.set('folds', 10)
    cores = psutil.cpu_count()
    GlobalConfig.set('cpu_count', cores)

    dataset = DatasetManager.KylbergTextures(num_classes=28, data_ratio=0.5)
    images = dataset.load_data()
    gc.collect()

    bm3d_images = []
    # Convert to BM3D images
    for image in images:
        new_image = BM3DELBP.BM3DELBPImage(image)
        bm3d_images.append(new_image)

    print("Image dataset loaded, loaded {} images".format(len(images)))

    noise_classifier = NoiseClassifier()
    cross_validator = dataset.get_cross_validator()

    bm3d_sigma = [10, 30, 40, 50]
    homomorphic_cutoff = [0.1, 0.5, 5, 10]
    homomorphic_a = [0.5, 0.75, 1.0]
    homomorphic_b = [0.1, 0.5, 1.0, 1.25]

    settings_jobs = []  # List of all configuration tuples

    for sigma_val in bm3d_sigma:
        for cutoff in homomorphic_cutoff:
            for a in homomorphic_a:
                for b in homomorphic_b:
                    settings_jobs.append((sigma_val, cutoff, a, b))

    results = []  # List of tuples (F1, sigma_val, cutoff, a, b)

    out_csv = os.path.join(GlobalConfig.get('CWD'), 'NoiseClassifierTuning', 'Results.txt')

    with Pool(processes=GlobalConfig.get('cpu_count'), maxtasksperchild=50) as pool:
        # Generate image featurevectors and replace DatasetManager.Image with BM3DELBP.BM3DELBPImage
        for result in tqdm.tqdm(pool.istarmap(do_classification, zip(settings_jobs, repeat(noise_classifier), repeat(cross_validator), repeat(bm3d_images))),
                                  total=len(settings_jobs), desc='NoiseClassifier tuning'):
            f1, sigma, cutoff, a, b = result
            # Log to CSV file
            if os.path.isfile(out_csv):
                # CSV exists, append to end of file
                with open(out_csv, 'a', encoding="utf-8", newline='') as resultsfile:
                    writer = csv.writer(resultsfile)
                    writer.writerow([f1, sigma, cutoff, a, b])

            else:
                # CSV does not exist. Write the headings
                with open(out_csv, 'w', encoding="utf-8", newline='') as resultsfile:
                    writer = csv.writer(resultsfile)
                    writer.writerow(['f1', 'sigma_psd', 'cutoff', 'a', 'b'])
                    writer.writerow([f1, sigma, cutoff, a, b])

            results.append(result)

    # Sort largest to smallest, by F1 score
    results.sort(key=lambda tup: tup[0], reverse=True)

    print("Finished tuning parameters.")
    print("The top 3 results were:")
    f1, sigma, cutoff, a, b = results[0]
    print("F1: {}, sigma_val: {}, cutoff_freq: {}, a: {}, b: {}".format(f1, sigma, cutoff, a, b))
    f1, sigma, cutoff, a, b = results[1]
    print("F1: {}, sigma_val: {}, cutoff_freq: {}, a: {}, b: {}".format(f1, sigma, cutoff, a, b))
    f1, sigma, cutoff, a, b = results[2]
    print("F1: {}, sigma_val: {}, cutoff_freq: {}, a: {}, b: {}".format(f1, sigma, cutoff, a, b))



if __name__ == '__main__':
    tune_noise_classifier()