# /usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: popunbom <fantom0779@gmail.com>
# Created At: 2019/06/22

import cv2
import numpy as np
from imgproc.segmentation import RegionSegmentation
from imgproc.edge import EdgeProcedures

seg = RegionSegmentation( "./img/resource/aerial_roi1_raw_denoised_clipped_ver2.png", logging=True )
# seg.get_segmented_image_with_label()
scores = print( seg.scores )
