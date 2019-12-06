# /usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: popunbom <fantom0779@gmail.com>
# Created At: 2019/10/14


"""
どこまで精度が出せるかどうか検証
"""
from os import path
from pprint import pprint
from itertools import product
from tqdm import tqdm

import json

import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils.assertion import NDARRAY_ASSERT, SAME_SHAPE_ASSERT, TYPE_ASSERT
from utils.logger import ImageLogger

plt.switch_backend("macosx")

from utils.common import eprint
from utils.evaluation import evaluation_by_confusion_matrix

PRECISION = 10


    
    
def find_threshold(src_img, ground_truth, logger=None):
    """
    閾値探索
    
    Parameters
    ----------
    src_img : numpy.ndarray
        入力画像
        - 8-Bit RGB カラー
    ground_truth : numpy.ndarray
        正解画像
        - 1-Bit (np.bool) 2値化画像
        - 黒：背景、白：被害領域
    logger : ImageLogger
        処理途中の画像をロギングする ImageLogger

    Returns
    -------
    dict, numpy.ndarray
        導き出された閾値と閾値処理を行った画像のタプル
    """
    
    NDARRAY_ASSERT(src_img, ndim=3, dtype=np.uint8)
    NDARRAY_ASSERT(ground_truth, ndim=2, dtype=np.bool)
    TYPE_ASSERT(logger, [None, ImageLogger])
    SAME_SHAPE_ASSERT(src_img, ground_truth, ignore_ndim=True)
    

    ground_truth = ground_truth.astype( bool )
    
    filter_params = dict(sp=40, sr=50)
    
    eprint("Pre-processing ... ", end="")
    img = cv2.pyrMeanShiftFiltering(src_img, **filter_params)
    eprint("done !")
    
    if logger:
        logger.logging_img(img, "filtered")
    
    if logger:
        logger.logging_dict(
            dict(
                type="cv2.pyrMeanShiftFiltering",
                **filter_params
            ),
            "filter_detail"
        )
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = [ img[:, :, i] for i in range(3) ]
    reasonable_thresh = {
        "Score": -1,
        "Range": -1
    }
    
    # Show Masked Histogram
    
    masked = img[ground_truth]
    
    for q_h_low, q_h_high, q_s_low, q_s_high, q_v_low, q_v_high in tqdm(list(product(np.linspace(50/PRECISION, 50, PRECISION), repeat=6))):
        # h_min, h_max = in_range_percentile(h.ravel(), q_h)
        # h_min, h_max = in_range_percentile(masked[:, 0], q_h)
        h_min, h_max = in_range_percentile(masked[:, 0], (q_h_low, q_h_high))
        # s_min, s_max = in_range_percentile(s.ravel(), q_s)
        # s_min, s_max = in_range_percentile(masked[:, 1], q_s)
        s_min, s_max = in_range_percentile(masked[:, 1], (q_s_low, q_s_high))
        # v_min, v_max = in_range_percentile(v.ravel(), q_v)
        # v_min, v_max = in_range_percentile(masked[:, 2], q_v)
        v_min, v_max = in_range_percentile(masked[:, 2], (q_v_low, q_v_high))
        
        
        _result = (
             ((h_min <= h) & (h <= h_max)) &
             ((s_min <= s) & (s <= s_max)) &
             ((v_min <= v) & (v <= v_max))
        )
        
        cm, metrics = evaluation_by_confusion_matrix(_result, ground_truth)
        
        if metrics["F Score"] > reasonable_thresh["Score"]:
            reasonable_thresh = {
                "Score": metrics["F Score"],
                "Confusion Matrix": cm,
                "Range": {
                    "H": (h_min, h_max, q_h_low, q_h_high),
                    "S": (s_min, s_max, q_s_low, q_s_high),
                    "V": (v_min, v_max, q_v_low, q_v_high),
                }
            }
            result = _result.copy()
    
    pprint(reasonable_thresh)
    
    if logger:
        logger.logging_dict(reasonable_thresh, "reasonal_thresh")
        logger.logging_img(result, "result")

    return reasonable_thresh, result


def find_morphology( src_result, ground_truth, logger=None ):
    """
    最適なモルフォロジー処理を模索
    
    Parameters
    ----------
    src_result : numpy.ndarray
        結果画像
        - 1-Bit (np.bool) 2値化画像
        - 黒：背景、白：被害領域
    ground_truth : numpy.ndarray
        正解画像
        - 1-Bit (np.bool) 2値化画像
        - 黒：背景、白：被害領域
    logger : ImageLogger
        処理途中の画像をロギングする ImageLogger

    Returns
    -------
    dict, numpy.ndarray
        導き出されたパラメータとモルフォロジー処理結果画像のタプル
    """
    
    NDARRAY_ASSERT( src_result, ndim=2, dtype=np.bool )
    NDARRAY_ASSERT( ground_truth, ndim=2, dtype=np.bool )
    TYPE_ASSERT( logger, [None, ImageLogger] )
    SAME_SHAPE_ASSERT( src_result, ground_truth, ignore_ndim=True )

    reasonable_params = {
        "Score" : {
            "Confusion Matrix": -1,
            "F Score": -1,
        },
        "Params": {
            "Operation" : "",
            "Kernel"    : {
                "Size"         : (-1, -1),
                "#_of_Neighbor": -1
            },
            "Iterations": -1
        }
    }
 
    src_result = (src_result * 255).astype( np.uint8 )
    
    for kernel_size in [3, 5]:
        for n_neighbor in [4, 8]:
            for operation in ["ERODE", "DILATE", "OPEN", "CLOSE"]:
                for n_iterations in range(1, 6):
                    
                    if n_neighbor == 4:
                        kernel = np.zeros(
                            (kernel_size, kernel_size),
                            dtype=np.uint8
                        )
                        kernel[kernel_size // 2, :] = 1
                        kernel[:, kernel_size // 2] = 1
                    else:
                        kernel = np.ones(
                            (kernel_size, kernel_size),
                            dtype=np.uint8
                        )
                    
                    _result = cv2.morphologyEx(
                        src_result,
                        cv2.__dict__[f"MORPH_{operation}"],
                        kernel=kernel,
                        iterations=n_iterations
                    ).astype(bool)

                    cm, metrics = evaluation_by_confusion_matrix( _result, ground_truth )
                    
                    eprint("Score:", metrics["F Score"])

                    if metrics["F Score"] > reasonable_params["Score"]["F Score"]:
                        
                        reasonable_params = {
                            "Score" : {
                                "Confusion Matrix": cm,
                                "F Score": metrics["F Score"],
                            },
                            "Params": {
                                "Operation" : operation,
                                "Kernel"    : {
                                    "Size"         : (kernel_size, kernel_size),
                                    "#_of_Neighbor": n_neighbor
                                },
                                "Iterations": n_iterations
                            }
                        }
                        
                        result = _result.copy()

    if logger:
        logger.logging_img(result, "result_morphology")
        logger.logging_dict(reasonable_params, "morph_detail")
    
        return reasonable_params, result
    

if __name__ == '__main__':
    # PATH_SRC_IMG = "img/resource/aerial_roi1_raw_ms_40_50.png"
    PATH_SRC_IMG = "img/resource/aerial_roi2_raw.png"
    PATH_GT_IMG = "img/resource/ground_truth/aerial_roi2.png"
    
    src_img = cv2.imread(
        PATH_SRC_IMG,
        cv2.IMREAD_COLOR
    )
    
    ground_truth = cv2.imread(
        PATH_GT_IMG,
        cv2.IMREAD_GRAYSCALE
    ).astype(bool)
    
    logger = ImageLogger(
        "./tmp/find_threshold",
        suffix=path.splitext(
            path.basename(
                PATH_SRC_IMG
            )
        )[0]
    )
    
    # _, smoothed = find_threshold(src_img, ground_truth, logger)

    src_result = cv2.imread(
        "./tmp/find_threshold/20191121_133309_aerial_roi2_raw/result.tiff",
        cv2.IMREAD_UNCHANGED
    ).astype(bool)
    
    _, fixed = find_morphology(src_result, ground_truth, logger)
