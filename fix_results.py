#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2020/01/19
# This is a part of EQDmgAnalyzr

import json
import re
from os import walk, listdir, path, makedirs
from os.path import splitext, join, exists, isdir
from textwrap import dedent

import cv2
import numpy as np

from damage_detector.building_damage import BuildingDamageExtractor
from imgproc.utils import imread_with_error, imwrite_with_error
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
    
    RESULT_ROOT_DIR = "./tmp/detect_building_damage/master-v2/fixed_histogram"
    
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

    RESULT_ROOT_DIR = "tmp/detect_building_damage/master-v2/fixed_histogram"
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
        
        result = imread_with_error(
            path.join(
                target_dir,
                "building_damage.tiff"
            ),
            cv2.IMREAD_UNCHANGED
        )
        
        # Fixing Image
        result_fixed = BuildingDamageExtractor._remove_tiny_area(
            (result * 255.0).astype(np.uint8)
        )
        
        
        imwrite_with_error(
            path.join(
                target_dir,
                "building_damage_fixed.tiff"
            ),
            result_fixed.astype(np.float32)
        )
        
        # Re-calc Score
        ground_truth = imread_with_error(
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


def generate_scores_each_gt_type():
    RESULT_ROOT_DIR = "tmp/detect_building_damage/master-v3/fixed_histogram/GT_BOTH"
    RESULT_GROUND_TRUTH = "img/resource/ground_truth"
    
    MAP_RESULT_IMAGE = {
        "meanshift_and_color_thresholding": "building_damage_fixed.tiff",
        "edge_angle_variance_with_hpf": "building_damage.tiff",
        "edge_pixel_classify": "building_damage.tiff"
    }
    
    C_RED = [0, 0, 255]
    C_ORANGE = [0, 127, 255]
    
    target_dirs = [
        path.join(
            RESULT_ROOT_DIR,
            entry
        )
        for entry in listdir(RESULT_ROOT_DIR)
        if isdir(
            path.join(
                RESULT_ROOT_DIR,
                entry
            )
        )
    ]
    
    for target_dir in target_dirs:
        eprint(
            f"Target: {target_dir}"
        )
        
        experiment_num, method_name = re.match(
            r".*/aerial_roi([0-9])_[0-9]{8}_[0-9]{6}_(.*)$",
            target_dir
        ).groups()
        
        eprint(dedent(f"""
            Experiment Num: {experiment_num}
            Method: {method_name}
        """))
        
        GT = cv2.imread(
            path.join(
                RESULT_GROUND_TRUTH,
                f"aerial_roi{experiment_num}.png"
            )
        )
        
        result = imread_with_error(
            path.join(
                target_dir,
                MAP_RESULT_IMAGE[method_name]
            ),
            cv2.IMREAD_UNCHANGED
        ).astype(bool)
        
        for gt_type in ["GT_BOTH", "GT_RED", "GT_ORANGE"]:
            if gt_type == "GT_BOTH":
                ground_truth = np.all(
                    (GT == C_RED) | (GT == C_ORANGE),
                    axis=2
                )
            elif gt_type == "GT_RED":
                ground_truth = np.all(
                    GT == C_RED,
                    axis=2
                )
            elif gt_type == "GT_ORANGE":
                ground_truth = np.all(
                    GT == C_ORANGE,
                    axis=2
                )
        
            cm, scores = evaluation_by_confusion_matrix(
                result,
                ground_truth
            )
            
            j = {
                "Confusion Matrix": cm,
                "Score"          : scores
            }
            
            save_path = path.join(
                target_dir,
                "evaluation",
                gt_type
            )
            
            if not exists(save_path):
                makedirs(save_path)
            
            json.dump(
                j,
                open(path.join(save_path, "scores.json"), "w"),
                ensure_ascii=False,
                sort_keys=True,
                indent="\t"
            )
        
        
if __name__ == '__main__':
    recalculate_scores()
    # remove_tiny_areas_and_recalc_score()
