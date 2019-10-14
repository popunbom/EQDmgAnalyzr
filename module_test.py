#/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: popunbom <fantom0779@gmail.com>
# Created At: 2019/09/23

import cv2
import numpy as np
from skimage.morphology import disk

from scipy import ndimage as ndi

from imgproc.edge import EdgeProcedures
from imgproc.hog import HoGFeature
from imgproc.utils import compute_by_window, zoom_to_img_size
from utils.mpl import show_images, imshow
from utils.logger import ImageLogger
from os import path

PATH_ROOT = "./img/resource"

img = cv2.imread(
    path.join( PATH_ROOT, "aerial_roi1_raw_denoised_clipped_ver2_equalized.png" )
    # path.join(PATH_ROOT, "aerial_roi2_strong-denoised_clipped_equalized.png")
    # "/Users/popunbom/Downloads/aerial_roi1_raw_denoised_test.png"
)
# X, Y, W, H = 0, 441, 348, 287
# img = img[Y:Y + H, X:X + W]
img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

logger = ImageLogger( "./img/tmp/module_test/hog+edge" )

hog = HoGFeature( img, logger )
hog.calc_features( feature_vector=False )

var = np.var(
    hog.features,
    axis=(2, 3, 4)
)

var = zoom_to_img_size( var / var.max(), img_gs.shape )

logger.logging_img( var, "hog_variance" )

edge = EdgeProcedures( img )

angle_var = edge.get_feature_by_window(
    edge.angle_variance_using_mean_vector,
    window_size=8,
    step=1
)

mag_stddev = edge.get_feature_by_window(
    edge.magnitude_stddev,
    window_size=8,
    step=1
)

logger.logging_img( angle_var, "angle_variance" )
logger.logging_img( mag_stddev, "magnitude_stddev" )

mag_stddev /= mag_stddev.max()
angle_var /= angle_var.max()

result = angle_var * mag_stddev

logger.logging_img( result, "result" )
