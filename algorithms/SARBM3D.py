import matlab.engine
import os
from scipy.io import savemat
import numpy as np
import cv2

from config import GlobalConfig

"""
The only existing SAR-BM3D Source Code on the internet is written in Matlab.
Furthermore, there is no source code! Only compiled binaries...
Fortunately, the Matlab Engine for Python exists, so I should be able to run them that way.
The SAR-BM3D executables can be downloaded here: http://www.grip.unina.it/web-download.html?dir=JSROOT/SAR-BM3D
"""


class SARBM3DFilter():
    def __init__(self, ecs=False):
        if ecs:
            self.SAMRBM3D_DIR = os.path.join('C:/', 'Local', 'SARBM3D_v10_win64')
        else:
            self.SAMRBM3D_DIR = os.path.join(os.getcwd(), 'algorithms', 'SARBM3D_v10_win64')

        # Make output folder if it doesn't exist
        self.OUT_DIR_PYTHON = os.path.join(self.SAMRBM3D_DIR, 'temp')
        if not (os.path.exists( self.OUT_DIR_PYTHON)):
            os.makedirs( self.OUT_DIR_PYTHON)

    def sar_bm3d_filter(self, image, image_name, L=50):
        """
        Performs SAR-BM3D filter on an ndarray. This is tricky because ndarrays cannot be natively passed to MATLAB.
        Instead, scipy.io.savemat is used to save the ndarray to a file, then the name of this file is passed to Matlab.
        Which opens the file, processes it, and returns it.
        :param image: 2D ndarray representing image to apply SAR-BM3D to
        :param image_name: Unique name of the image
        :param L: L parameter for SAR-BM3D filter. L=50 did not sacrifice too much texture
        :return: 2D ndarray representing image after filtering
        """
        width, height = image.shape
        out_file = '{}.mat'.format(image_name)
        out_file_python = os.path.join(self.OUT_DIR_PYTHON, out_file)
        out_file_matlab = os.path.join('temp', out_file)

        image = image.astype(np.double)  # The MATLAB SAR-BM3D filter wants the image encoded as a double.
        savemat(out_file_python, {'image_data': image})  # Encode the numpy ndarray as a MATLAB .mat file
        FILTERED_IMAGE = self.eng.SARBM3D_Python_Helper(out_file_matlab, L)
        FILTERED_IMAGE = np.array(FILTERED_IMAGE._data)  # Convert from mlarray.double into numpy ndarray
        FILTERED_IMAGE = FILTERED_IMAGE.astype(np.float32)  # Convert back to float32
        FILTERED_IMAGE = FILTERED_IMAGE.reshape((width, height), order='F')  # Reshape into original width and height
        # Rescale back to [-1, 1]
        FILTERED_IMAGE = cv2.normalize(FILTERED_IMAGE, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        os.remove(out_file_python)  # Delete the .mat file
        return FILTERED_IMAGE

    def connect_matlab(self):
        # Initialise MATLAB engine
        self.eng = matlab.engine.start_matlab()
        # Check if SARBM3D_v10_win64 executables folder exists
        if not os.path.exists(self.SAMRBM3D_DIR):
            raise FileNotFoundError('SARBM3D_v10_win64 executables missing. Place them in: ' + self.SAMRBM3D_DIR)
        # Switch to directory
        self.eng.cd(self.SAMRBM3D_DIR, nargout=0)

    def disconnect_matlab(self):
        self.eng.quit()