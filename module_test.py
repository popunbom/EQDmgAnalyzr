#/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: popunbom <fantom0779@gmail.com>
# Created At: 2019/09/23

import cv2
import numpy as np
from skimage.filters import gaussian

from imgproc.edge import EdgeProcedures
from utils.logger import ImageLogger

img = cv2.imread(
    "img/resource/aerial_roi1_raw_denoised_clipped_v3.png",
    # "img/resource/label.bmp",
    cv2.IMREAD_COLOR
)

logger = ImageLogger(
    base_path="./tmp/module_test",
    prefix="edge_angle_variance"
)

e = EdgeProcedures(img, ksize=3, algorithm="sobel")

_, e.edge_magnitude = cv2.threshold(
    (e.edge_magnitude * 255).astype(np.uint8),
    thresh=0,
    maxval=255,
    type=cv2.THRESH_OTSU
)
e.edge_magnitude = (e.edge_magnitude / 255.0).astype(np.float32)

logger.logging_img(
    e.edge_magnitude,
    "magnitude"
)

logger.logging_img(
    e.edge_angle,
    "angle",
    cmap="jet"
)

logger.logging_img(
    e.get_angle_colorized_img(),
    "angle_colorized"
)
