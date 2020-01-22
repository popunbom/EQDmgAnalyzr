# -*- coding: utf-8 -*-
# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2019-09-25

# This is a part of EQDmgAnalyzr

from os import path
import cv2
from utils.evaluation import evaluation_by_confusion_matrix

if __name__ == '__main__':
    
    # RESULT_ROOT_PATH = "./img/result"
    
    RESULT_ROOT_PATH = "./tmp/notebooks/EdgeAngleVariance+HPF"
    GT_ROOT_PATH = "./img/resource/ground_truth"
    
    result = cv2.imread(
        path.join( RESULT_ROOT_PATH, "20191009_152809/System - Result(Merged).png" )
    )
    
    ground_truth = cv2.imread(
        path.join( GT_ROOT_PATH, "IMG_6955-qv.png" )
    )
    
    confusion_matrix, metrics = evaluation_by_confusion_matrix( result, ground_truth )
    
    for k, v in confusion_matrix.items():
        print( f"{k}: {v}" )
    
    for k, v in metrics.items():
        print( f"{k:15s}: {v:.4f}" )
