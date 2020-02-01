#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2020-01-21
# This is a part of EQDmgAnalyzr
from os import path
from textwrap import dedent

import cv2

import numpy as np
import pymeanshift

from imgproc.utils import imread_with_error
from utils.assertion import NDARRAY_ASSERT, SAME_SHAPE_ASSERT
from utils.common import eprint


def gen_ground_truth_by_type(ground_truth, gt_type):
    C_RED = [0, 0, 255]
    C_ORANGE = [0, 127, 255]

    if gt_type == "GT_BOTH":
        return np.all(
            (ground_truth == C_RED) | (ground_truth == C_ORANGE),
            axis=2
        )
    elif gt_type == "GT_RED":
        return np.all(
            ground_truth == C_RED,
            axis=2
        )
    elif gt_type == "GT_ORANGE":
        return np.all(
            ground_truth == C_ORANGE,
            axis=2
        )
    else:
        return None


def calc_hsv_metrics_by_ground_truth(img, ground_truth, gt_type="GT_ORANGE"):
    NDARRAY_ASSERT(img, ndim=3, dtype=np.uint8)
    SAME_SHAPE_ASSERT(img, ground_truth, ignore_ndim=True)
    
    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )
    
    ground_truth = gen_ground_truth_by_type(
        ground_truth,
        gt_type
    )
    
    h, s, v = [
        hsv[:, :, i][ground_truth == True]
        for i in range(3)
    ]
    
    metrics = {
        k: {
            "min": ch.min(),
            "max": ch.max(),
            "mean": np.mean(ch),
            "median": np.median(ch),
            "stddev": np.std(ch),
            "var": np.var(ch)
        }
        for k, ch in {
            "H": h,
            "S": s,
            "V": v
        }.items()
    }
    
    return metrics


def calc_hsv_mean_each_image():
    ROOT_DIR_SRC = "./img/resource/aerial_image/fixed_histogram_v2"
    ROOT_DIR_GT = "./img/resource/ground_truth"
    
    # experiments = [1, 2, 3, 4, 5, 6]
    experiments = [5]
    
    params_mean_shift = {
        # "spatial_radius": 8,
        # "range_radius": 5,
        "spatial_radius": 8,
        "range_radius": 5,
        "min_density": 0
    }
    
    gt_type = "GT_ORANGE"
    
    results = dict()
    
    for exp_num in experiments:
    
        src_img = imread_with_error(
            path.join(
                ROOT_DIR_SRC,
                f"aerial_roi{exp_num}.png"
            )
        )
        
        ground_truth = imread_with_error(
            path.join(
                ROOT_DIR_GT,
                f"aerial_roi{exp_num}.png"
            )
        )
        
        eprint(dedent(f"""
            Experiment Num: {exp_num}
                   gt_type: {gt_type}
        """))
        
        eprint(f"Do Mean-Shift ... ", end="")
        src_img = pymeanshift.segment(
            src_img,
            **params_mean_shift
        )[0]
        eprint("done")
        
        metrics = calc_hsv_metrics_by_ground_truth(
            src_img,
            ground_truth
        )
        
        # print(dedent(f"""
        #     Mean (H): {means['H']}
        #     Mean (S): {means['S']}
        #     Mean (V): {means['V']}
        # """))
        
        results[
            f"aerial_roi{exp_num}"
        ] = metrics
    
    
    for exp_name, metrics in results.items():
    
        print("\t".join(["EXP_NAME", exp_name]))
        print("\t".join(["Ch.", *list(metrics.values())[0].keys()]))
        for ch_name, metric in metrics.items():
            # for k, v in results.items():
            #     print(",".join([
            #         k,
            #         *metric.values()
            #     ]))
            print("\t".join([
                str(x)
                for x in [ch_name, *metric.values()]
            ]))
    return results

if __name__ == '__main__':
    calc_hsv_mean_each_image()
