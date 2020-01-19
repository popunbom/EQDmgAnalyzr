#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2020/01/19
# This is a part of EQDmgAnalyzr
import re
from os import walk, listdir
from os.path import join, isdir, splitext
from pprint import pprint
from textwrap import dedent

import cv2
import numpy as np
from matplotlib import cm

from utils.assertion import TYPE_ASSERT, SAME_SHAPE_ASSERT, SAME_NDIM_ASSERT, NDARRAY_ASSERT
from utils.common import eprint

C_RED = [0, 0, 255]
C_ORANGE = [0, 127, 255]

C_GREEN = [0, 255, 0]
C_MAGENTA = [255, 0, 255]
C_BLACK = [0, 0, 0]
C_WHITE = [255, 255, 255]

CONT = 0.3

ROOT_DIR_RESULT = "./tmp/detect_building_damage/master/fixed_histogram"
ROOT_DIR_GT = "./img/resource/ground_truth"
ROOT_DIR_SRC = "./img/resource/aerial_image/fixed_histogram"


def hsv_blending(bg_img, fg_img):
    NDARRAY_ASSERT(fg_img, ndim=3)
    SAME_SHAPE_ASSERT(bg_img, fg_img)
    
    if bg_img.ndim == 3:
        cv2.cvtColor(
            bg_img,
            cv2.COLOR_BGR2GRAY
        )
    if bg_img.ndim == 2:
        cv2.cvtColor(
            bg_img,
            cv2.COLOR_GRAY2BGR
        )
        
    bg_hsv, fg_hsv = [
        cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for img in [bg_img, fg_img]
    ]
    
    b_h, b_s, b_v = [bg_hsv[:, :, i] for i in range(3)]
    f_h, f_s, f_v = [fg_hsv[:, :, i] for i in range(3)]
    
    dst = cv2.cvtColor(
        np.dstack(
            [f_h, f_s, b_v]
        ),
        cv2.COLOR_HSV2BGR
    )
    
    return dst
    
    
def merge_arrays_by_mask(array_1, array_2, mask):
    NDARRAY_ASSERT(mask, ndim=2, dtype=np.bool)
    SAME_NDIM_ASSERT(array_1, array_2)
    SAME_SHAPE_ASSERT(array_1, array_2)
    SAME_SHAPE_ASSERT(array_1, mask, ignore_ndim=True)
    
    array_1 = array_1.copy()
    array_2 = array_2.copy()
    
    Z = np.zeros(
        (1 if array_1.ndim == 2 else array_1.shape[2],),
        dtype=array_1.dtype
    )

    array_1[mask == True] = Z
    array_2[mask == False] = Z
    
    return array_1 + array_2
    

def imread_with_error(file_name, flags=cv2.IMREAD_COLOR):
    img = cv2.imread(file_name, flags)
    
    if img is None:
        raise FileNotFoundError(
            file_name
        )
    
    return img


def write_images(root_path, filenames_and_images):
    
    for file_name, img in filenames_and_images:
        base_name = splitext(file_name)[0]
        ext = ".png"
        if img.dtype != np.uint8:
            ext = ".tiff"
        
        save_path = join(
            root_path,
            base_name + ext
        )
        
        result = cv2.imwrite(
            save_path,
            img
        )
        
        if result:
            eprint(
                f"Saved Image -- {save_path}"
            )
        else:
            raise RuntimeError(
                "Failed to save image"
            )
    

result_dirs = sum([
    [
        join(
            ROOT_DIR_RESULT,
            gt_type,
            d
        )
        for d in listdir(
            join(
                ROOT_DIR_RESULT,
                gt_type
            )
        )
        if isdir(
            join(
                ROOT_DIR_RESULT,
                gt_type,
                d
            )
        )
    ]
    for gt_type in ["GT_BOTH", "GT_ORANGE", "GT_RED"]
], [])

result_dirs = sorted(result_dirs)
pprint(result_dirs[1:2])

for result_dir in result_dirs[1:2]:
    gt_type, experiment_num = re.match(
        r".*/GT_(.*)/aerial_roi([0-9]).*",
        result_dir
    ).groups()

    eprint(dedent(f"""
        Experiment Num: {experiment_num}
        GT_TYPE: GT_{gt_type}
    """))

    # Load: ground_truth
    ground_truth = imread_with_error(
        join(
            ROOT_DIR_GT,
            f"aerial_roi{experiment_num}.png"
        )
    )
    
    # Load: source
    src = imread_with_error(
        join(
            ROOT_DIR_SRC,
            f"aerial_roi{experiment_num}.png"
        )
    )
    src_gs = cv2.cvtColor(
        cv2.cvtColor(
            src,
            cv2.COLOR_BGR2GRAY
        ),
        cv2.COLOR_GRAY2BGR
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
    
    ground_truth = (ground_truth * 255).astype(np.uint8)
    
    Z = np.zeros(
        ground_truth.shape[:2],
        dtype=np.uint8
    )

    if "meanshift_and_color_thresholding" in result_dir:
        result = (imread_with_error(
            join(
                result_dir,
                "building_damage_fixed.tiff"
            ),
            cv2.IMREAD_UNCHANGED
        ) * 255).astype(np.uint8)
        
        confusion_matrix = np.dstack(
            [result, ground_truth, result]
        )
        
        missing = confusion_matrix.copy()
        missing[~np.all(missing == C_GREEN, axis=2)] = [0, 0, 0]
        
        wrong = confusion_matrix.copy()
        wrong[~np.all(wrong == C_MAGENTA, axis=2)] = [0, 0, 0]
        
        # Extract
        missing_extracted = merge_arrays_by_mask(
            (src_gs * CONT).astype(np.uint8),
            src,
            np.all(missing == C_GREEN, axis=2)
        )
        missing_extracted_with_color = hsv_blending(
            src_gs,
            missing
        )
        wrong_extracted = merge_arrays_by_mask(
            (src_gs * CONT).astype(np.uint8),
            src,
            np.all(wrong == C_MAGENTA, axis=2)
        )
        wrong_extracted_with_color = hsv_blending(
            src_gs,
            wrong
        )
        
        
        write_images(
            result_dir,
            [
                ("confusion_matrix", confusion_matrix),
                ("missing", missing),
                ("wrong", wrong),
                ("missing_extracted", missing_extracted),
                ("missing_extracted_with_color", missing_extracted_with_color),
                ("wrong_extracted", wrong_extracted),
                ("wrong_extracted_with_color", wrong_extracted_with_color)
            ]
        )
        
    
    elif "edge_angle_variance_with_hpf" in result_dir:
        result = (imread_with_error(
            join(
                result_dir,
                "building_damage.tiff"
            ),
            cv2.IMREAD_UNCHANGED
        ) * 255).astype(np.uint8)

        fd_angle_var = imread_with_error(
            join(
                result_dir,
                "edge_angle_variance/angle_variance.png"
            ),
        )
        
        fd_hpf = imread_with_error(
            join(
                result_dir,
                "high_pass_filter/HPF_gray.tiff"
            ),
            cv2.IMREAD_UNCHANGED
        )
        fd_hpf = (cm.get_cmap("jet")(
            fd_hpf / fd_hpf.max()
        ) * 255).astype(np.uint8)[:, :, [2, 1, 0]]
        
        # Generate Image of Confusion Matrix
        confusion_matrix = np.dstack(
            [result, ground_truth, result]
        )
    
        missing = confusion_matrix.copy()
        missing[~np.all(missing == C_GREEN, axis=2)] = [0, 0, 0]
    
        wrong = confusion_matrix.copy()
        wrong[~np.all(wrong == C_MAGENTA, axis=2)] = [0, 0, 0]
    
        # Extract
        missing_extracted = merge_arrays_by_mask(
            (src_gs * CONT).astype(np.uint8),
            src,
            np.all(missing == C_GREEN, axis=2)
        )
        missing_extracted_with_color = hsv_blending(
            src_gs,
            missing
        )
        missing_extracted_by_anglevar = merge_arrays_by_mask(
            (src_gs * CONT).astype(np.uint8),
            fd_angle_var,
            np.all(missing == C_GREEN, axis=2)
        )
        missing_extracted_by_hpf = merge_arrays_by_mask(
            (src_gs * CONT).astype(np.uint8),
            fd_hpf,
            np.all(missing == C_GREEN, axis=2)
        )
        wrong_extracted = merge_arrays_by_mask(
            (src_gs * CONT).astype(np.uint8),
            src,
            np.all(wrong == C_MAGENTA, axis=2)
        )
        wrong_extracted_with_color = hsv_blending(
            src_gs,
            wrong
        )
        wrong_extracted_by_anglevar = merge_arrays_by_mask(
            (src_gs * CONT).astype(np.uint8),
            fd_angle_var,
            np.all(wrong == C_MAGENTA, axis=2)
        )
        wrong_extracted_by_hpf = merge_arrays_by_mask(
            (src_gs * CONT).astype(np.uint8),
            fd_hpf,
            np.all(wrong == C_MAGENTA, axis=2)
        )
        
    
        write_images(
            result_dir,
            [
                ("confusion_matrix", confusion_matrix),
                ("missing", missing),
                ("wrong", wrong),
                ("missing_extracted", missing_extracted),
                ("missing_extracted_with_color", missing_extracted_with_color),
                ("missing_extracted_by_anglevar", missing_extracted_by_anglevar),
                ("missing_extracted_by_hpf", missing_extracted_by_hpf),
                ("wrong_extracted", wrong_extracted),
                ("wrong_extracted_with_color", wrong_extracted_with_color),
                ("wrong_extracted_by_anglevar", wrong_extracted_by_anglevar),
                ("wrong_extracted_by_hpf", wrong_extracted_by_hpf)
            ]
        )


    elif "edge_pixel_classify" in result_dir:
        pass
