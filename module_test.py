# /usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: popunbom <fantom0779@gmail.com>
# Created At: 2019/09/23
from time import sleep

import cv2
import numpy as np
from scipy.signal import gaussian
from skimage.feature import canny

from imgproc.edge import EdgeProcedures
from imgproc.edge_line_feature import EdgeLineFeatures
from imgproc.utils import compute_by_window
from utils.logger import ImageLogger

img = cv2.imread(
    # "img/edge_sample.png",
    "img/resource/aerial_roi1_raw_denoised_clipped.png",
    # "img/resource/label.bmp",
    cv2.IMREAD_COLOR
)

img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )

logger = ImageLogger(
    base_path="./tmp/module_test",
    # prefix="edge_line_feature"
    prefix="edge_angle_variance"
)


ep = EdgeProcedures( img )

fd_img = ep.get_feature_by_window(
    ep.angle_variance_using_mean_vector,
    window_size=33,
    step=1
)
# fd_img = compute_by_window(
#     (ep.edge_magnitude, ep.edge_angle),
#     ep.angle_variance_using_mean_vector,
#     window_size=8,
#     step=1,
# )

logger.logging_img( fd_img, "fd_img" )
logger.logging_img( fd_img, "fd_colorized", cmap="jet" )


# fd = EdgeLineFeatures(img, logger=logger)
#
# canny_params = dict(
#     sigma=0.1,
#     low_threshold=0.2,
#     high_threshold=0.5
# )
#
# fd.do_canny(**canny_params)
#
# classified = fd.classify()
#
# fd.calc_metrics()
#
# classified = cv2.imread(
#     "./tmp/module_test/edge_line_feature_20191214_170535/classified.png",
#     cv2.IMREAD_GRAYSCALE
# )
#
#
# def _f_calc_percentage(roi):
#     LABELS = EdgeLineFeatures.LABELS
#     BG = LABELS["BG"]
#     ENDPOINT = LABELS["endpoint"]
#     BRANCH = LABELS["branch"]
#
#
#     n_edges = roi.size - roi[roi == BG].size
#     # n_edges = roi.size
#
#     if n_edges == 0:
#         return 0
#
#     n_endpoint = roi[roi == ENDPOINT].size
#     n_branch = roi[roi == BRANCH].size
#
#     return (n_endpoint + n_branch) / n_edges
#
# params = {
#     "_f_calc_weighted_percentage": {
#         "sigma": 8.0
#     }
# }
#
# def _f_calc_weighted_percentage(roi):
#     LABELS = EdgeLineFeatures.LABELS
#     BG = LABELS["BG"]
#     ENDPOINT = LABELS["endpoint"]
#     BRANCH = LABELS["branch"]
#
#     sigma = params["_f_calc_weighted_percentage"]["sigma"]
#
#     gaussian_kernel = np.outer(
#         gaussian(roi.shape[0], std=sigma),
#         gaussian(roi.shape[1], std=sigma)
#     )
#
#     # w_edges = np.sum(gaussian_kernel) - np.sum(gaussian_kernel[roi == BG])
#     w_edges = np.sum(gaussian_kernel)
#
#     if w_edges == 0:
#         return 0
#
#     w_endpoint = np.sum(gaussian_kernel[roi == ENDPOINT])
#     w_branch = np.sum(gaussian_kernel[roi == BRANCH])
#
#     return (w_endpoint + w_branch) / w_edges
#
#
# # for ws in [4, 8, 16, 32]:
# for ws in [5, 9, 17, 33]:
#     fd_img = compute_by_window(
#         imgs=classified,
#         # func=_f_calc_percentage,
#         func=_f_calc_weighted_percentage,
#         window_size=ws,
#         step=1,
#         n_worker=12
#     )
#     logger.logging_img(fd_img, f"fd_ws{ws}_st1")
#     logger.logging_img(fd_img, f"fd_ws{ws}_st1_colorize", cmap="jet")
#
# logger.logging_dict(params, "params_window_proc")
#
# for _ in range(10):
#     sleep(1)
#     print("\a")
