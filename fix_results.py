#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2020/01/19
# This is a part of EQDmgAnalyzr

import json
import re
from os import walk, listdir, path
from os.path import splitext, join, exists, isdir
from textwrap import dedent

import cv2
import numpy as np

from damage_detector.building_damage import BuildingDamageExtractor
from utils.common import eprint
from utils.evaluation import evaluation_by_confusion_matrix, calculate_metrics

def _add_prefix(file_path, prefix, delimiter="_"):
    base_name, ext = splitext(file_path)
    
    return f"{base_name}{delimiter}{prefix}{ext}"


def fix_result_format():
    RESULT_ROOT_DIR = "./tmp/detect_building_damage/2020_01_09_whole_test"
    
    EXP_PREFIX = "aerial_roi"
    
    OPTIONS = [
        "no_norm",
        "GT_BOTH",
    ]
    
    dirs = [
        path.join(RESULT_ROOT_DIR, d)
        for d in listdir(RESULT_ROOT_DIR)
        if isdir(path.join(RESULT_ROOT_DIR, d))
           and "meanshift_and_color_thresholding" in d
    ]
    
    for d in dirs:
        json_path = path.join(
            d,
            "params_finder/params_morphology.json"
        )
        
        if exists(json_path):
            j = json.load(open(json_path))
            
            j = {
                **j,
                "Score"           : j["Score"]["F Score"],
                "Confusion Matrix": j["Score"]["Confusion Matrix"]
            }
            
            json.dump(
                j,
                open(
                    _add_prefix(json_path, "fixed"),
                    "w"
                ),
                ensure_ascii=False,
                sort_keys=True,
                indent="\t"
            )
            eprint(f"Fixed: {json_path}")
        else:
            eprint(f"Not exists: {json_path}")


def recalculate_scores():
    
    RESULT_ROOT_DIR = "./tmp/detect_building_damage/master/fixed_histogram"
    
    def add_prefix(file_path, prefix, delimiter="_"):
        base_name, ext = splitext(file_path)
        
        return f"{base_name}{delimiter}{prefix}{ext}"
    
    
    json_paths = sum([
        [
            (dir_path, file_name)
            for file_name in file_names
            if "json" in file_name
        ]
        for (dir_path, dir_names, file_names) in walk(RESULT_ROOT_DIR)
    ], [])
    
    for dir_path, dir_name in json_paths:
        
        json_path = join(dir_path, dir_name)
        
        if exists(json_path):
            j = json.load(open(json_path))
            
            if "Confusion Matrix" in j:
                
                j = {
                    **j,
                    "Score": calculate_metrics(j["Confusion Matrix"])
                }
            
            json.dump(
                j,
                open(
                    add_prefix(json_path, "fixed"),
                    "w"
                ),
                ensure_ascii=False,
                sort_keys=True,
                indent="\t"
            )
            eprint(f"Fixed: {json_path}")
        else:
            eprint(f"Not exists: {json_path}")


def remove_tiny_areas_and_recalc_score():

    RESULT_ROOT_DIR = "tmp/detect_building_damage/master/fixed_histogram"
    RESULT_GROUND_TRUTH = "img/resource/ground_truth"
    
    C_RED = [0, 0, 255]
    C_ORANGE = [0, 127, 255]
    
    target_dirs = [
        dir_path
        for (dir_path, dir_names, file_names) in walk(RESULT_ROOT_DIR)
        if dir_path.endswith("meanshift_and_color_thresholding")
    ]
    
    for target_dir in target_dirs:
        eprint(
            f"Target: {target_dir}"
        )
        
        gt_type, experiment_num = re.match(
            r".*/GT_(.*)/aerial_roi([0-9]).*",
            target_dir
        ).groups()
        
        eprint(dedent(f"""
            Experiment Num: {experiment_num}
            GT_TYPE: GT_{gt_type}
        """))
        
        result = cv2.imread(
            path.join(
                target_dir,
                "building_damage.tiff"
            ),
            cv2.IMREAD_UNCHANGED
        )
        if result is None:
            eprint("Image load error.")
        
        # Fixing Image
        result_fixed = BuildingDamageExtractor._remove_tiny_area(
            (result * 255.0).astype(np.uint8)
        )
        
        cv2.imwrite(
            path.join(
                target_dir,
                "building_damage_fixed.tiff"
            ),
            result_fixed.astype(np.float32)
        )
        
        # Re-calc Score
        
        ground_truth = cv2.imread(
            path.join(
                RESULT_GROUND_TRUTH,
                f"aerial_roi{experiment_num}.png"
            )
        )
        
        if gt_type == "BOTH":
            ground_truth = np.all(
                (ground_truth == C_RED) | (ground_truth == C_ORANGE),
                axis=2
            )
        elif gt_type == "RED":
            ground_truth = np.all(
                ground_truth == C_RED,
                axis=2
            )
        elif gt_type == "ORANGE":
            ground_truth = np.all(
                ground_truth == C_ORANGE,
                axis=2
            )
        
        cm, scores = evaluation_by_confusion_matrix(
            result_fixed,
            ground_truth
        )
        
        j = {
            "Confusion Matrix": cm,
            "Score"          : scores
        }
        
        json.dump(
            j,
            open(
                path.join(
                    target_dir,
                    "scores_fixed_result.json"
                ),
                "w"
            ),
            ensure_ascii=False,
            sort_keys=True,
            indent="\t"
        )
        
if __name__ == '__main__':
    remove_tiny_areas_and_recalc_score()
