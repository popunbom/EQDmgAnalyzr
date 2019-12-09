# /usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: popunbom <fantom0779@gmail.com>
# Created At: 2019/09/23

import cv2
import numpy as np
from skimage.feature import canny
from skimage.filters import gaussian

from imgproc.edge import EdgeProcedures
from imgproc.edge_line_feature import EdgeLineFeatures
from imgproc.utils import compute_by_window
from utils.logger import ImageLogger

img = cv2.imread(
    "img/resource/aerial_roi1_raw_denoised_clipped.png",
    # "img/resource/label.bmp",
    cv2.IMREAD_COLOR
)

img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )

logger = ImageLogger(
    base_path="./tmp/module_test",
    prefix="edge_line_feature"
)


# ep = EdgeProcedures( img )
#
# fd_img = ep.get_feature_by_window(
#     ep.angle_variance_using_mean_vector,
#     window_size=8,
#     step=1
# )
# fd_img = compute_by_window(
#     (ep.edge_magnitude, ep.edge_angle),
#     ep.angle_variance_using_mean_vector,
#     window_size=8,
#     step=1,
# )
#
# logger.logging_img( fd_img, "fd_img" )
# logger.logging_img( fd_img, "fd_colorized", cmap="jet" )


fd = EdgeLineFeatures(img, logger=logger)

canny_params = dict(
    sigma=0.1,
    low_threshold=0.2,
    high_threshold=0.5
)

fd.do_canny(**canny_params)

classified = fd.classify()

fd.calc_metrics()

# classified = cv2.imread(
#     "./tmp/module_test/edge_line_feature_20191208_140316/classified.png",
#     cv2.IMREAD_GRAYSCALE
# )

# def calc_percentage(roi):
#     LABELS = EdgeLineFeatures.LABELS
#     BG = LABELS["BG"],
#     ENDPOINT = LABELS["endpoint"]
#     BRANCH = LABELS["branch"]
#     PASSING = LABELS["passing"]
#
#     # n_edges = roi.size - roi[roi == BG].size
#     n_edges = roi.size
#
#     if n_edges == 0:
#         return 0
#
#     n_endpoint = roi[roi == ENDPOINT].size
#     n_branch = roi[roi == BRANCH].size
#
#     return (n_endpoint + n_branch) / n_edges
#
#
# for ws in [4, 8, 16, 32]:
#     fd_img = compute_by_window(
#         imgs=classified,
#         func=calc_percentage,
#         window_size=ws,
#         step=1
#     )
#     logger.logging_img(fd_img, f"fd_ws{ws}_st1")
#     logger.logging_img(fd_img, f"fd_ws{ws}_st1_colorize", cmap="jet")
#
