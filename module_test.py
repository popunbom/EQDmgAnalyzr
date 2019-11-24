#/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: popunbom <fantom0779@gmail.com>
# Created At: 2019/09/23

import cv2
import numpy as np

from imgproc.edge_line_feature import EdgeLineFeatures
from utils.logger import ImageLogger


img = cv2.imread(
    "img/resource/edge2.bmp",
    cv2.IMREAD_GRAYSCALE
)

logger = ImageLogger(
    "./tmp/edge_line_features"
)

inst = EdgeLineFeatures(
    img,
    logger=logger
)

inst.classify_pixel()
inst.calc_metrics()
