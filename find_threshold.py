# /usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: popunbom <fantom0779@gmail.com>
# Created At: 2019/10/14


"""
どこまで精度が出せるかどうか検証
"""
from os import path
from time import sleep

import cv2
import numpy as np

from utils.common import eprint
from utils.evaluation import evaluation_by_confusion_matrix

PRECISION = 10 ** -2

RESULT_DIR = "img/tmp/module_test/hog+edge/20191014_183809"

FILE_NAMES = [
    "angle_variance.tiff",
    "hog_variance.tiff",
    "magnitude_stddev.tiff",
    "result.tiff",
]

ground_truth = cv2.imread(
    "img/resource/ground_truth/aerial_roi1.png",
    cv2.IMREAD_GRAYSCALE
)

ground_truth = ground_truth.astype( bool )

for file in FILE_NAMES:
    file_path = path.join( RESULT_DIR, file )
    
    eprint( f"Load Image: {file_path}" )
    
    result = cv2.imread(
        file_path,
        cv2.IMREAD_UNCHANGED
    )
    
    if result.dtype != np.uint8 and result.ndim != 2:
        eprint( f"ERROR! dtype={result.dtype}, ndim={result.ndim}" )
    
    reasonable_thresh = {
        "Score": -1,
        "Range": (-1, -1)
    }
    
    for th_1 in np.arange( 0, result.max(), PRECISION ):
        for th_2 in np.arange( th_1, result.max(), PRECISION ):
            _, metrix = evaluation_by_confusion_matrix(
                (th_1 < result) & (result < th_2),
                ground_truth
            )
            score = metrix["F Score"]
            
            eprint( f"\r{(th_1, th_2)}: {score:.4f}", end="" )
            sleep( 0.1 )
            
            if score > reasonable_thresh["Score"]:
                reasonable_thresh["Score"] = score
                reasonable_thresh["Range"] = (th_1, th_2)
    
    eprint()
    print( reasonable_thresh )
