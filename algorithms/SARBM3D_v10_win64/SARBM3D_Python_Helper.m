%SARBM3D Python Helper.
% This is required because Python ndarrays cannot be natively passed into
% Matlab, so instead my Python code saves the ndarray with
% scipy.io.savemat, then passes the filename into this function.

function IMG_FILTERED = SARBM3D_Python_Helper(path, L)
%SARBM3D_filter Takes an image path, loads the image and applies
%    SARBM3D filter.
%   Detailed explanation goes here
Z = load(path); 
IMG = Z.image_data;
IMG_FILTERED = SARBM3D_v10(IMG,L);

end

