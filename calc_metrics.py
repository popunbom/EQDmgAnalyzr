#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2020-01-15
# This is a part of EQDmgAnalyzr


import json
import re
import sys
from os import listdir, path
from os.path import isdir, exists
from pathlib import Path
from textwrap import dedent
import pyperclip

from utils.common import eprint

RESULT_ROOT_DIR = "./tmp/detect_building_damage/2020_01_09_whole_test"

EXP_PREFIX = "aerial_roi"

OPTIONS = [
    "no_norm",
    "GT_BOTH",
]

dirs = [d for d in listdir(RESULT_ROOT_DIR) if isdir(path.join(RESULT_ROOT_DIR, d))]

def add_prefix(file_path, prefix, delimiter="_"):
    base_name, ext = path.splitext(file_path)
    
    return f"{base_name}{delimiter}{prefix}{ext}"

def recalc(path_json):
    
    eprint(f"Re-calculate metrics: {path_json}")
    
    j = json.load(open(path_json))
    
    if isinstance(j["Score"], float):
        cm = j["Confusion Matrix"]
    
    FN, FP, TN, TP = cm["FN"], cm["FP"], cm["TN"], cm["TP"]
    
    metrics = {
        "Accuracy"   : 0 if (TP + FP + FN + TN) == 0 else 100 * (TP + TN) / (TP + FP + FN + TN),
        "Recall"     : 0 if (TP + FN) == 0 else 100 * TP / (TP + FN),
        "Specificity": 0 if (FP + TN) == 0 else 100 * TN / (FP + TN),
        "Precision"  : 0 if (TP + FP) == 0 else 100 * TP / (TP + FP),
    }
    
    j["Score"] = {
        **metrics,
        "F Score": j["Score"]
    }

    json.dump(
        j,
        open(
            add_prefix(path_json, "with_metrics"),
            "w"
        ),
        ensure_ascii=False,
        sort_keys=True,
        indent="\t"
    )
    
    return

for d in dirs:
    base_dir = path.join(
        RESULT_ROOT_DIR,
        d,
    )
    paths_json = []
    
    if "meanshift_and_color_thresholding" in d:
        paths_json = [
            path.join(base_dir, "params_finder/params_morphology.json"),
            path.join(base_dir, "params_finder/color_thresholds_in_hsv.json")
        ]
    elif "edge_pixel_classify" in d:
        paths_json = [
            path.join(base_dir, "params_finder_edge_line_feature/params.json")
        ]
    elif "edge_angle_variance_with_hpf" in d:
        paths_json = [
            path.join(base_dir, "params_finder/params_subtracted_thresholds.json"),
            path.join(base_dir, "params_finder_angle_variance/params.json")
        ]

    for path_json in paths_json:
        if exists(path_json):
            recalc(path_json)
        else:
            eprint(
                f"File is not exists !: {path_json}"
            )
