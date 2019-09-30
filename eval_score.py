# -*- coding: utf-8 -*-
# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2019-09-25

# This is a part of EQDmgAnalyzr

from os import path
import cv2
import numpy as np

ROOT_PATH = "./img/result"

result = cv2.imread(
    path.join( ROOT_PATH, "roi1_result.png" )
)[:, :, 1].astype( bool )

gt = cv2.imread(
    path.join( ROOT_PATH, "roi1_ground_truth.png" )
)[:, :, 2].astype( bool )

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
    print(f"{k}: {v:2.3f}")
