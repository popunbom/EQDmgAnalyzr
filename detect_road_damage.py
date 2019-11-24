#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2019-11-19
# This is a part of EQDmgAnalyzr

import cv2
import numpy as np
from os import path

from skimage.morphology import skeletonize
from skimage.filters import sobel

import matplotlib.pyplot as plt

from utils.assertion import TYPE_ASSERT, SAME_SHAPE_ASSERT, NDARRAY_ASSERT
from utils.logger import ImageLogger

plt.switch_backend( "macosx" )


def detect_road_damage( result, road_mask, logger=None ):
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
    
    NDARRAY_ASSERT( result, ndim=2, dtype=np.bool )
    NDARRAY_ASSERT( road_mask, ndim=2, dtype=np.bool )
    SAME_SHAPE_ASSERT( result, road_mask )
    
    # 道路マスクに Sobel → 細線化 を適用する
    skel = skeletonize(
        sobel( road_mask ).astype( bool ),
        method="zhang"
    )
    
    if logger:
        logger.logging_img( skel, "skeletonize" )
    
    # 建物被害結果から距離画像を作成する
    dist = cv2.distanceTransform(
        ( result * 255 ).astype(np.uint8),
        cv2.DIST_L2,
        maskSize=5
    )
    
    if logger:
        logger.logging_img( dist, "distance", cmap="jet" )
    
    # Ver.1: 道路境界線上の距離値のみを抽出する
    # scores = dist * skel.astype( np.float32 )
    # TODO: 2値化でOKか？
    # scores[scores > 0] = 1
    
    # Ver.2: 距離画像を道路マスクでマスキング
    
    scores = dist * road_mask
    
    scores = ((scores / scores.max()) * 255).astype(np.uint8)
    
    if logger:
        logger.logging_img( scores, "scores" )
        logger.logging_img( scores, "scores_visualize", cmap="jet" )
    
    plt.imshow( scores, cmap="jet" )
    plt.show()
    
    return scores


if __name__ == '__main__':
    
    DIR_ROAD_MASK = "./img/resource/road_mask"
    DIR_RESULT = "./tmp/find_threshold/20191121_165941_aerial_roi2_raw"
    
    road_mask = cv2.imread(
        path.join( DIR_ROAD_MASK, "aerial_roi2.png" ),
        cv2.IMREAD_GRAYSCALE
    ).astype(bool)
    
    result = cv2.imread(
        path.join(
            DIR_RESULT, "result_morphology.tiff"
        ),
        cv2.IMREAD_UNCHANGED
    ).astype(bool)
    
    logger = ImageLogger(
        base_path="./tmp/detect_road_damage"
    )
    
    detect_road_damage( result, road_mask, logger )
