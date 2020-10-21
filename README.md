# COMP3200-Independent-Project-Texture-Classification
Source code for experiments for Individual Project Dissertation.

This project seeks to evaluate the performance of a number of texture classification algorithms, RLBP (Robust Local Binary Patterns), MRLBP (Multiresolution Local Binary Patterns), MRELBP (Median Robust Extended Local Binary Patterns) and BM3DELBP (Block Matching and 3D filtering Extended Local Binary Pattern).

These algorithms take a dataset with a series of classes of textures, then attempt classify unseen textures into these texture classes.

The project tests classification accuracy when trained on ordinary textures and tested on textures with a variety of transformations including a different resolution (scale), various types of noise (gaussian, salt & pepper and speckle) and different rotations, which can be configured by CLI arguments. 

This allows us to test algorithms for invariance to different transformations.


### Installation:

1. `git pull https://github.com/OscarVanL/COMP3200-Texture-Classification`
2. Download [Kylberg Texture Dataset]( http://www.cb.uu.se/~gustaf/texture/) 'Texture dataset *without* rotated patches' and unzip them to `data/kylberg`
3. Download [Kylberg Texture Dataset]( http://www.cb.uu.se/~gustaf/texture/) 'Texture dataset *with* rotated patches' and unzip them to `data/kylberg-rotated`
4. Download MATLAB executables for [SAR-BM3D filter v1.0](http://www.grip.unina.it/web-download.html?dir=JSROOT/SAR-BM3D) and unzip them to `algorithms/SARBM3D_v10_win64`. Note:  I have only tested this with the Windows 10 executables.
5. Install the Visual C++ 2010 Redistributable Package (x64) as described in SAR-BM3D's README.txt
6. Install MATLAB and install the [MATLAB Engine API for Python](https://uk.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).
7. Install dependencies `pip install requirements.txt`

### Operation:

Run beginClassification.py with the desired arguments.

##### For example: 

`python beginClassification.py -a MRELBP -d kylberg -t 0.8 -s 0.5 -n gaussian -i 10 -r -m` 
`-a MRELBP` : Use the MRELBP algorithm to extract features and classify

`-d kylberg` : Use the Kylberg texture dataset

`-t 0.8` : Use a train:test ratio of 80:20

`-s 0.5` : Load the Kylberg textures at 50% scale (large scales result in slow computation)

`-n gaussian` : Nose to apply to testing images

`-i 10` : Intensity of noise. For gaussian this is the standard deviation σ.

`-r` : Load rotations for the test images

`-m` : Use multi-core procesing

##### Args:

`-a` or `--algorithm` : Configure the Algorithm to use. ('RLBP', 'MRLBP', 'MRELBP', 'BM3DELBP', NoiseClassifier')

`-d` or `--dataset` : Configure the Dataset to use (Currently only 'kylberg')

`-t` or `--train-ratio` : Ratio of the dataset to use for training. Eg: 0.8

`-s` or `--scale` : Amount to rescale the training images. Eg: 0.5 to halve the image_scaled resolution

`-S` or `--test-scale` : Amount to rescale the testing images. This is used for the scale invariance test

`-k` or `--folds` : Number of cross folds to complete

`-r` or `--rotations` : Whether to rotate ecah image_scaled in the dataset 12 times

`-n` or `--noise` : Which type of noise to apply ('gaussian', 'speckle', 'salt-pepper')

`-i` or `--noise-intensity` : How much noise to apply (Sigma / Variance / Ratio)

`-m` or `--multiprocess` : Whether to use multi process featurevector generation

`-e` or `--example` : Generate example images used in dissertation report

`--data-ratio` : Ratio of the dataset to load (eg: 0.5 will only load 50% of the dataset's textures per class)

`--mrlbp-classifier` : Which classifier to use for MRLBP ('knn' or 'svm')

`--noise-train` : Apply noise to the training images too

`--ecs` : If running on an ECS Labs machine, loads the dataset and SAR-BM3D from C:\Local instead of CWD.

`--debug` : Whether to run in debug mode (uses a reduced dataset to speed up execution, prints more stuff)


### Kylberg Texture Dataset: 
G. Kylberg. The Kylberg Texture Dataset v. 1.0, Centre for Image Analysis,
Swedish University of Agricultural Sciences and Uppsala University,
External report (Blue series) No. 35.

Available online at: http://www.cb.uu.se/~gustaf/texture/

### SAR-BM3D Texture Filter:

S.Parrilli, M.Poderico, C.V.Angelino, L.Verdoliva. A nonlocal SAR image denoising algorithm based on LLMMSE wavelet shrinkage, *IEEE Transactions on Geoscience and Remote Sensing*, vol.50, no.2, pp.606-616, Feb. 2012.

Available online at: http://www.grip.unina.it/web-download.html?dir=JSROOT/SAR-BM3D
