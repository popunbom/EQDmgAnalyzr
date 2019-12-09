# /usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2019-10-21

# This is a part of EQDmgAnalyzr

import cv2
import numpy as np
from skimage.morphology import disk

from scipy import ndimage as ndi

from imgproc.edge import EdgeProcedures
from imgproc.glcm import GLCMFeatures
from imgproc.hog import HoGFeature
from imgproc.preproc import denoise_by_median_and_meanshift
from imgproc.utils import compute_by_window, zoom_to_img_size
from utils.common import eprint
from utils.mpl import show_images, imshow
from utils.logger import ImageLogger
from os import path

import matplotlib.pyplot as plt

# plt.switch_backend("macosx")

PATH_ROOT = "./img/resource"

img = cv2.imread(
    # path.join( PATH_ROOT, "label.bmp" )
    path.join( PATH_ROOT, "aerial_roi1_raw_denoised_clipped.png" )
    # path.join(PATH_ROOT, "aerial_roi2_strong-denoised_clipped_equalized.png")
    # "/Users/popunbom/Downloads/roof.png"
)

# eprint( "Pre-processing ... " )
# img = cv2.pyrMeanShiftFiltering( img, sp=40, sr=50 )
img = denoise_by_median_and_meanshift(img)

gt = cv2.imread( "img/resource/ground_truth/aerial_roi1.png", cv2.IMREAD_GRAYSCALE )
gt = gt.astype( np.bool )

logger = ImageLogger( "./img/tmp/eval_system" )

glcm = GLCMFeatures( img, logger=logger )

glcm_feature = glcm.calc_features( "dissimilarity" )

edge = EdgeProcedures( img, algorithm="sobel", ksize=3 )

angle_var = edge.get_feature_by_window(
    edge.angle_variance_using_mean_vector,
    window_size=8,
    step=2
)

mag_stddev = edge.get_feature_by_window(
    edge.magnitude_stddev,
    window_size=8,
    step=2
)

logger.logging_img( img, "source" )

angle_var = zoom_to_img_size( angle_var, img.shape )
mag_stddev = zoom_to_img_size( mag_stddev, img.shape )

angle_col = edge.get_angle_colorized_img( max_intensity=True )
edge_mag = edge._calc_magnitude()

logger.logging_img( edge_mag, "edge_magnitude" )
logger.logging_img( angle_var, "angle_variance" )
logger.logging_img( mag_stddev, "magnitude_stddev" )

# plt.figure("Source")
# plt.imshow( img[:, :, [2, 1, 0]] )
#
# plt.figure( "Magnitude" )
# plt.imshow( edge_mag, cmap="jet" )
logger.logging_img( edge_mag, "Magnitude", cmap="jet" )
#
# plt.figure( "Angle (colorized)" )
# plt.imshow( angle_col, cmap="jet" )
logger.logging_img( angle_col, "Angle_colorized", cmap="jet" )
#
# plt.figure( "Angle Variance" )
# plt.imshow( angle_var, cmap="jet" )
logger.logging_img( angle_var, "Angle_variance", cmap="jet" )
#
# plt.figure( "Magnitude StdDev" )
# plt.imshow( mag_stddev, cmap="jet" )
#
# plt.figure( "Histogram of Angle Variance" )
# plt.hist( angle_var[gt == True], bins=1000, color="b", alpha=0.5, log=True )
# plt.hist( angle_var[gt == False], bins=1000, color="r", alpha=0.4, log=True )
#
# plt.figure( "Histogram of Magnitude StdDev" )
# plt.hist( mag_stddev[gt == True], bins=1000, color="b", alpha=0.5, log=True )
# plt.hist( mag_stddev[gt == False], bins=1000, color="r", alpha=0.4, log=True )
#
# plt.show()
