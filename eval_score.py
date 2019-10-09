# -*- coding: utf-8 -*-
# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2019-09-25

# This is a part of EQDmgAnalyzr

from os import path
import cv2
import numpy as np

# RESULT_ROOT_PATH = "./img/result"

RESULT_ROOT_PATH = "./tmp/notebooks/EdgeAngleVariance+HPF"
GT_ROOT_PATH = "./img/resource/ground_truth"

result = cv2.imread(
    path.join( RESULT_ROOT_PATH, "20191009_152809/System - Result(Merged).png" )
)

gt = cv2.imread(
    path.join( GT_ROOT_PATH, "IMG_6955-qv.png" )
)

assert result.shape[:2] == gt.shape[:2], \
    f"""
    ERROR! -- result must be same size of ground-truth
    result.shape: {result.shape}
        gt.shape: {gt.shape}
    """

for data in [result, gt]:
    if data.ndim == 3:
        data = np.any(data > 0, axis=2)


TP = np.count_nonzero(
    result & gt
)
FP = np.count_nonzero(
    result & (gt == False)
)
FN = np.count_nonzero(
    (result == False) & gt
)
TN = np.count_nonzero(
    (result == False) & (gt == False)
)

scores = {
    "True Positive" : TP,
    "False Positive": FP,
    "False Negative": FN,
    "True Negative" : TN,
    "Accuracy"      : 100 * (TP + TN) / (TP + FP + FN + TN),
    "Recall"        : 100 * TP / (TP + FN),
    "Specificity": 100 * TN / (FP + TN),
    "Precision": 100 * TP / (TP + FP),
}
scores.update(
    {
        "F Score": (2 * scores["Recall"] * scores["Precision"]) / (scores["Recall"] + scores["Precision"])
    }
)


for k, v in scores.items():
    print(f"{k:15s}: {v:2.3f}")
