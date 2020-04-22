import matlab.engine
import os
from scipy.io import savemat, loadmat
import numpy as np
import cv2

"""
The only existing SAR-BM3D Source Code on the internet is written in Matlab.
Furthermore, there is no source code! Only compiled binaries...
Fortunately, the Matlab Engine for Python exists, so I should be able to run them that way.
The SAR-BM3D executables can be downloaded here: http://www.grip.unina.it/web-download.html?dir=JSROOT/SAR-BM3D
"""


class SARBM3DFilter():
    def __init__(self):
        # Initialise MATLAB engine
        self.eng = matlab.engine.start_matlab()
        # Check of SARBM3D_v10_win64 executables folder exists
        self.SAMRBM3D_DIR = os.path.join(os.getcwd(), 'algorithms', 'SARBM3D_v10_win64')
        if not os.path.exists(self.SAMRBM3D_DIR):
            raise FileNotFoundError('SARBM3D_v10_win64 executables missing. Place them in: ' + self.SAMRBM3D_DIR)
        # Switch to directory
        self.eng.cd(self.SAMRBM3D_DIR, nargout=0)
        print("MATLAB Working DIR:", self.eng.pwd())
        if self.eng.isprime(37):
            print("MATLAB connection successful")

    def sar_bm3d_filter(self, image, image_name, L=50):
        """
        Performs SAR-BM3D filter on an ndarray.
        This is tricky because ndarrays cannot be natively passed to MATLAB.
        Instead, scipy.io.savemat is used to save the ndarray to a file, then the name of this file is passed to Matlab.
        Which opens the file, processes it, and returns it.
        See https://stackoverflow.com/a/45284125/6008271
        :param image: 2D ndarray representing image to apply SAR-BM3D to
        :param image_name: Unique name of the image
        :param L: L parameter for SAR-BM3D filter. L=50 did not sacrifice too much texture
        :return: 2D ndarray representing image after filtering
        """
        width, height = image.shape

        # Scale in range [0, 255]:
        scaled_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        out_file = '{}.mat'.format(image_name)
        out_dir_python = os.path.join('algorithms', 'SARBM3D_v10_win64', 'temp')
        # Make output folder if it doesn't exist
        if not (os.path.exists(out_dir_python)):
            os.makedirs(out_dir_python)

        out_file_python = os.path.join(out_dir_python, out_file)
        out_file_matlab = os.path.join('temp', out_file)

        # The MATLAB SAR-BM3D filter wants the image encoded as a double.
        image = image.astype(np.double)
        # Encode the numpy ndarray as a MATLAB .mat file
        savemat(out_file_python, {'image_data': image})
        FILTERED_IMAGE = self.eng.SARBM3D_Python_Helper(out_file_matlab, L)
        # Convert from mlarray.double into numpy ndarray
        FILTERED_IMAGE = np.array(FILTERED_IMAGE._data)
        # Convert back to float32
        FILTERED_IMAGE = FILTERED_IMAGE.astype(np.float32)
        # Reshape into original width and height
        FILTERED_IMAGE = FILTERED_IMAGE.reshape((width, height), order='F')

        # Rescale back to [-1, 1]
        FILTERED_IMAGE = cv2.normalize(FILTERED_IMAGE, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # Delete the .mat file
        os.remove(out_file_python)
        return FILTERED_IMAGE

    def disconnect_matlab(self):
        self.eng.quit()