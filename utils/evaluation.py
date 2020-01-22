# /usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: popunbom <fantom0779@gmail.com>
# Created At: 2019/10/14

"""
utils/evaluation.py: 精度評価に関係する処理
"""

import numpy as np

from utils.assertion import SAME_SHAPE_ASSERT, NDARRAY_ASSERT


def calculate_metrics( confusion_matrix ):
    cm = confusion_matrix
    
    TP, FP, FN, TN = cm["TP"], cm["FP"], cm["FN"], cm["TN"]
    
    metrics = {
        "Accuracy"    : 0 if (TP + FP + FN + TN) == 0 else 100 * (TP + TN) / (TP + FP + FN + TN),
        "Recall"      : 0 if (TP + FN) == 0 else 100 * TP / (TP + FN),
        "Specificity" : 0 if (FP + TN) == 0 else 100 * TN / (FP + TN),
        "Precision"   : 0 if (TP + FP) == 0 else 100 * TP / (TP + FP),
        # 見落とし率(Missing-Rate): `FN / (TP+FN)`
        "Missing-Rate": 0 if (TP + FN) == 0 else 100 * FN / (TP + FN),
        # 誤り率(Wrong-Rate): `FP / (TP+FP)`
        "Wrong-Rate"  : 0 if (TP + FP) == 0 else 100 * FP / (TP + FP),
    }
    
    metrics.update(
        {
            "F Score": 0 if (metrics["Recall"] + metrics["Precision"]) == 0 else (2 * metrics["Recall"] * metrics[
                "Precision"]) / (metrics["Recall"] + metrics["Precision"])
        }
    )
    
    return metrics


def evaluation_by_confusion_matrix( result, ground_truth ):
    """
    混同行列 (Confusion-Matrix) を用いた
    精度評価
    
    Notes
    -----
    - `result` と `ground_truth` は bool 型で
      同じサイズの行列である必要がある

    Parameters
    ----------
    result : numpy.ndarray
        出力結果
    ground_truth : numpy.ndarray
        正解データ

    Returns
    -------
    confusion_matrix, metrics : tuple
        混同行列と各種スコアの tuple
    """
    NDARRAY_ASSERT( result, dtype=bool )
    NDARRAY_ASSERT( ground_truth, dtype=bool )
    SAME_SHAPE_ASSERT( result, ground_truth, ignore_ndim=True )
    
    for data in [result, ground_truth]:
        if data.ndim == 3:
            data = np.any( data, axis=2 )
    
    TP = np.count_nonzero(
        result & ground_truth
    )
    FP = np.count_nonzero(
        result & (ground_truth == False)
    )
    FN = np.count_nonzero(
        (result == False) & ground_truth
    )
    TN = np.count_nonzero(
        (result == False) & (ground_truth == False)
    )
    
    confusion_matrix = {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
    }
    
    metrics = calculate_metrics(confusion_matrix)
    
    return confusion_matrix, metrics
