import os
import cv2
from data import ImageUtils, DatasetManager
from algorithms import RLBP, MRELBP, BM3DELBP, NoiseClassifier, SARBM3D
from config import GlobalConfig


class GenerateExamples:
    def __init__(self, path):
        """
        Generate Example images for dissertation write-up
        :param path: Image to produce example images with
        """
        self.image_path = path
        image_name = path.split(os.sep)[-1].partition('.')[0]
        #image_uint8 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        image_uint8 = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (0, 0),
                                fx=0.5,
                                fy=0.5)
        # Convert from uint8 to float32 without normalizing to zero mean
        image_unscaled = ImageUtils.convert_uint8_image_float32(image_uint8)
        # Convert from uint8 to float32 while normalizing to zero mean
        image_scaled = ImageUtils.scale_uint8_image_float32(image_uint8)
        image_gauss_10 = ImageUtils.add_gaussian_noise_skimage(image_scaled, 10)
        image_gauss_25 = ImageUtils.add_gaussian_noise_skimage(image_scaled, 25)
        image_speckle_002 = ImageUtils.add_speckle_noise_skimage(image_scaled, 0.02)
        image_speckle_004 = ImageUtils.add_speckle_noise_skimage(image_scaled, 0.04)
        image_salt_pepper_002 = ImageUtils.add_salt_pepper_noise_skimage(image_scaled, 0.02)
        image_salt_pepper_004 = ImageUtils.add_salt_pepper_noise_skimage(image_scaled, 0.04)
        image_label = path.split(os.sep)[-1].partition('-')[0]
        # Generate different permutations of this sample image
        self.image_uint8 = DatasetManager.Image(image_uint8, image_name, image_label)
        self.image_unscaled = DatasetManager.Image(image_unscaled, image_name, image_label)
        self.image_scaled = DatasetManager.Image(image_scaled, image_name, image_label)
        self.image_gauss_10 = DatasetManager.Image(image_gauss_10, image_name, image_label)
        self.image_gauss_10.test_noise='gaussian'; self.image_gauss_10.test_noise_val=10
        self.image_gauss_25 = DatasetManager.Image(image_gauss_25, image_name, image_label)
        self.image_gauss_25.test_noise = 'gaussian'; self.image_gauss_25.test_noise_val = 25
        self.image_speckle_002 = DatasetManager.Image(image_speckle_002, image_name, image_label)
        self.image_speckle_002.test_noise = 'speckle'; self.image_speckle_002.test_noise_val = 0.02
        self.image_speckle_004 = DatasetManager.Image(image_speckle_004, image_name, image_label)
        self.image_speckle_004.test_noise = 'speckle'; self.image_speckle_004.test_noise_val = 0.04
        self.image_salt_pepper_002 = DatasetManager.Image(image_salt_pepper_002, image_name, image_label)
        self.image_salt_pepper_002.test_noise = 'salt-pepper'; self.image_salt_pepper_002.noise_val = 0.02
        self.image_salt_pepper_004 = DatasetManager.Image(image_salt_pepper_004, image_name, image_label)
        self.image_salt_pepper_004.test_noise = 'salt-pepper'; self.image_salt_pepper_004.noise_val = 0.04
        self.path = os.path.join(GlobalConfig.get('CWD'), 'example')

        write_image(ImageUtils.convert_float32_image_uint8(self.image_unscaled.data), None, image_name + '-unedited.png')
        write_image(ImageUtils.convert_float32_image_uint8(self.image_scaled.data), None, image_name + '-scaled.png')

    def write_noise_examples(self):
        """
        Produce examples of images with noise types applied
        :return: None
        """
        print("Producing Noisy Image examples for:", self.image_scaled.name)
        write_image(ImageUtils.convert_float32_image_uint8(self.image_gauss_10.data),
                         'Noise Applied', self.image_scaled.name + '-Gaussian-Sigma-10.png')
        write_image(ImageUtils.convert_float32_image_uint8(self.image_gauss_25.data),
                         'Noise Applied', self.image_scaled.name + '-Gaussian-Sigma-25.png')
        write_image(ImageUtils.convert_float32_image_uint8(self.image_speckle_002.data),
                         'Noise Applied', self.image_scaled.name + '-Speckle-Var-0.02.png')
        write_image(ImageUtils.convert_float32_image_uint8(self.image_speckle_004.data),
                         'Noise Applied', self.image_scaled.name + '-Speckle-Var-0.04.png')
        write_image(ImageUtils.convert_float32_image_uint8(self.image_salt_pepper_002.data),
                         'Noise Applied', self.image_scaled.name + '-Salt-Pepper-2%.png')
        write_image(ImageUtils.convert_float32_image_uint8(self.image_salt_pepper_004.data),
                         'Noise Applied', self.image_scaled.name + '-Salt-Pepper-4%.png')
        print("Finished producing Noisy Image examples")

    def write_RLBP_example(self):
        print("Producing RLBP example for:", self.image_scaled.name)
        rlbp = RLBP.RobustLBP(save_img=True)
        rlbp.describe(self.image_scaled, test_image=False)
        print("Finished producing RLBP example")

    def write_MRLBP_example(self):
        print("Producing MRLBP examples for:", self.image_scaled.name)
        mrlbp = RLBP.MultiresolutionLBP(p=[8, 16, 24], r=[1, 2, 3], save_img=True)
        mrlbp.describe(self.image_scaled, test_image=False)
        print("Finished producing MRLBP examples")

    def write_MRELBP_example(self):
        print("Producing MRELBP examples for:", self.image_scaled.name)
        mrelbp = MRELBP.MedianRobustExtendedLBP(r1=[2, 4, 6, 8], p=8, w_center=3, w_r1=[3, 5, 7, 9], save_img=True)
        mrelbp.describe(self.image_scaled, test_image=False)
        print("Finished producing MRELBP examples")

    def write_BM3DELBP_example(self):
        print("Producing BM3DELBP examples for:", self.image_scaled.name)
        bm3delbp = BM3DELBP.BM3DELBP(save_img=True)
        bm3delbp.describe(self.image_scaled.data, test_image=False)
        bm3delbp_noise_classifier = NoiseClassifier.NoiseClassifier(save_img=True)
        print("No noise")
        bm3delbp_noise_classifier.describe(self.image_scaled, test_image=False)
        print("Gaussian Sigma 10")
        bm3delbp_noise_classifier.describe(self.image_gauss_10, test_image=False)
        print("Gaussian Sigma 25")
        bm3delbp_noise_classifier.describe(self.image_gauss_25, test_image=False)
        print("Speckle 0.02")
        bm3delbp_noise_classifier.describe(self.image_speckle_002, test_image=False)
        print("Speckle 0.02")
        bm3delbp_noise_classifier.describe(self.image_speckle_004, test_image=False)
        print("Salt & Pepper 2%")
        bm3delbp_noise_classifier.describe(self.image_salt_pepper_002, test_image=False)
        print("Salt & Pepper 4%")
        bm3delbp_noise_classifier.describe(self.image_salt_pepper_004, test_image=False)
        # Also demonstrate SAR-BM3D filter for speckle noise
        print("SAR-BM3D Example")
        sarbm3d = SARBM3D.SARBM3DFilter(ecs=False)
        sarbm3d.connect_matlab()
        sarbm3d_filtered_002 = sarbm3d.sar_bm3d_filter(ImageUtils.convert_float32_image_uint8(self.image_speckle_002.data), self.image_speckle_002.name, L=4)
        sarbm3d_filtered_004 = sarbm3d.sar_bm3d_filter(ImageUtils.convert_float32_image_uint8(self.image_speckle_004.data), self.image_speckle_004.name, L=4)
        sarbm3d.disconnect_matlab()
        write_image(ImageUtils.convert_float32_image_uint8(sarbm3d_filtered_002), os.path.join('BM3DELBP', 'NoiseFilters'),
                    self.image_speckle_002.name + '-Speckle-Var-0.02-SARBM3D-Filtered.png')
        write_image(ImageUtils.convert_float32_image_uint8(sarbm3d_filtered_004), os.path.join('BM3DELBP', 'NoiseFilters'),
                    self.image_speckle_004.name + '-Speckle-Var-0.04-SARBM3D-Filtered.png')


def write_image(image, folder, image_name):
    if folder is None:
        out_dir = os.path.join(os.getcwd(), 'example')
    else:
        out_dir = os.path.join(os.getcwd(), 'example', folder)

    # If requested folder doesn't exist, make it
    try:
        os.makedirs(out_dir)
    except FileExistsError:
        pass

    out_file = os.path.join(out_dir, image_name)
    
    print("out_file: ", out_file)

    cv2.imwrite(out_file, image)
