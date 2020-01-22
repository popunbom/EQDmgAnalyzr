#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2019-11-19
# This is a part of EQDmgAnalyzr


import platform

import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import path

from utils.assertion import SAME_SHAPE_ASSERT, NDARRAY_ASSERT
from utils.logger import ImageLogger


if platform.system() == "Darwin":
    plt.switch_backend("macosx")


def detect_road_damage_1(result, road_mask, logger=None):
    """
    建物被害の結果から道路上の被害抽出を行う
    
    Parameters
    ----------
    result : numpy.ndarray
        建物被害の結果
        黒を背景、白を被害領域として
        2値化されている必要がある
    
    road_mask : numpy.ndarray
        道路マスク画像
        黒を背景、白を道路領域として
        2値化されている必要がある
    
    logger : ImageLogger, default is None
        処理途中の画像をロギングする ImageLogger
    
    Returns
    -------
    numpy.ndarray
        道路上の被害抽出結果
        
    Notes
    -----
    - `result` と `road_mask` は同じ大きさである必要がある
    """
    
    NDARRAY_ASSERT(result, ndim=2, dtype=np.bool)
    NDARRAY_ASSERT(road_mask, ndim=2, dtype=np.bool)
    SAME_SHAPE_ASSERT(result, road_mask)
    
    # 1. 建物被害結果に道路マスクを適用
    
    result_extracted = result * road_mask
    result_extracted = (result_extracted * 255).astype(np.uint8)
    
    if logger:
        logger.logging_img(result_extracted, "result_extracted")
    
    # 2. 1. の画像から距離画像を作成する
    dist = cv2.distanceTransform(
        result_extracted,
        cv2.DIST_L2,
        maskSize=5
    )
    # FIXED: Normalize
    # dist = (dist / dist.max() * 255).astype(np.uint8)
    
    if logger:
        logger.logging_img(dist, "distance", cmap="gray")
        logger.logging_img(dist, "distance_visualized", cmap="jet")
    
    return dist


def detect_road_damage_2(result, road_mask, logger=None):
    """
    建物被害の結果から道路上の被害抽出を行う
    
    Parameters
    ----------
    result : numpy.ndarray
        建物被害の結果
        黒を背景、白を被害領域として
        2値化されている必要がある
    
    road_mask : numpy.ndarray
        道路マスク画像
        黒を背景、白を道路領域として
        2値化されている必要がある
    
    logger : ImageLogger, default is None
        処理途中の画像をロギングする ImageLogger
    
    Returns
    -------
    numpy.ndarray
        道路上の被害抽出結果
        
    Notes
    -----
    - `result` と `road_mask` は同じ大きさである必要がある
    """
    
    NDARRAY_ASSERT(result, ndim=2, dtype=np.bool)
    NDARRAY_ASSERT(road_mask, ndim=2, dtype=np.bool)
    SAME_SHAPE_ASSERT(result, road_mask)
    
    result = (result * 255).astype(np.uint8)
    
    dist = cv2.distanceTransform(
        result,
        cv2.DIST_L2,
        maskSize=5
    )
    # FIXED: Normalize
    # dist = (dist / dist.max() * 255).astype(np.uint8)
    
    if logger:
        logger.logging_img(dist, "distance", cmap="gray")
        logger.logging_img(dist, "distance_visualized", cmap="jet")
    
    result_extracted = dist * road_mask
    
    if logger:
        logger.logging_img(result_extracted, "result_extracted")
        
    return result_extracted


if __name__ == '__main__':
    
    DIR_ROAD_MASK = "./img/resource/road_mask"
    FILE_ROAD_MASK = "aerial_roi1.png"
    
    # PATH_RESULT = "./tmp/find_threshold/20191115_134949_aerial_roi1_raw_denoised_clipped/result_morphology.tiff"
    # PATH_RESULT = "./tmp/find_threshold/20191121_165941_aerial_roi2_raw/result_morphology.tiff"
    # PATH_RESULT = "./tmp/notebooks/EdgeAngleVariance+HPF/20190924_224939/Thresh - Sub(AngleVar, HPF).tiff"
    PATH_RESULT = "./tmp/notebooks/EdgeAngleVariance+HPF/20191206_144540/System - Result(Merged).png"
    
    road_mask = cv2.imread(
        path.join(DIR_ROAD_MASK, FILE_ROAD_MASK),
        cv2.IMREAD_GRAYSCALE
    ).astype(bool)
    
    result = cv2.imread(
        PATH_RESULT,
        cv2.IMREAD_UNCHANGED
    ).astype(bool)
    
    logger = ImageLogger(
        base_path="./tmp/detect_road_damage"
    )
    
    # detect_road_damage_1(result, road_mask, logger)
    detect_road_damage_2(result, road_mask, logger)
