#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2020/01/19
# This is a part of EQDmgAnalyzr
import json
import re
from os import walk, listdir, makedirs
from os.path import join, isdir, splitext, basename, dirname, exists
from pprint import pprint
from textwrap import dedent

import cv2
import numpy as np
from matplotlib import cm

from imgproc.utils import imread_with_error, imwrite_with_error
from utils.assertion import TYPE_ASSERT, SAME_SHAPE_ASSERT, SAME_NDIM_ASSERT, NDARRAY_ASSERT
from utils.common import eprint

from itertools import combinations

# from imgproc import overlay
from utils.evaluation import evaluation_by_confusion_matrix

C_RED = [0, 0, 255]
C_ORANGE = [0, 127, 255]

C_GREEN = [0, 255, 0]
C_MAGENTA = [255, 0, 255]
C_BLACK = [0, 0, 0]
C_WHITE = [255, 255, 255]

CONT = 0.3

# ROOT_DIR_RESULT = "./tmp/detect_building_damage/master-v2/fixed_histogram"
ROOT_DIR_RESULT = "./tmp/detect_building_damage/change_window_size"
ROOT_DIR_GT = "./img/resource/ground_truth"
ROOT_DIR_SRC = "./img/resource/aerial_image/CLAHE_with_WB_adjust"
ROOT_DIR_ROAD_MASK = "./img/resource/road_mask"
ROOT_DIR_VEG_MASK = "./img/resource/vegetation_mask"


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



def write_images(root_path, filenames_and_images, prefix="study"):
    
    for file_name, img in filenames_and_images:
        base_name = splitext(file_name)[0]
        ext = ".png"
        if img.dtype != np.uint8:
            ext = ".tiff"
        
        save_path = join(
            root_path,
            "_".join([
                prefix,
                base_name + ext
            ])
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


def gen_study_data():
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
        # for gt_type in ["GT_BOTH", "GT_ORANGE", "GT_RED"]
        for gt_type in ["GT_BOTH"]
    ], [])
    
    result_dirs = sorted(result_dirs)
    pprint(result_dirs[1:2])
    
    for result_dir in result_dirs:
    
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
                    "edge_angle_variance/angle_variance.tiff"
                ),
                cv2.IMREAD_UNCHANGED
            )
            fd_angle_var = (cm.get_cmap("jet")(
                fd_angle_var / fd_angle_var.max()
            ) * 255).astype(np.uint8)[:, :, [2, 1, 0]]
            
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
            
            # Overlay
            fd_overlay_angle_var = hsv_blending(
                src_gs,
                fd_angle_var
            )
            # fd_overlay_angle_var = overlay.do_gimp_overlay(
            #     src_gs, fd_angle_var, overlay.FUNC_GRAIN_MERGE
            # )
            fd_overlay_hpf = hsv_blending(
                src_gs,
                fd_hpf
            )
            # fd_overlay_hpf = overlay.do_gimp_overlay(
            #     src_gs, fd_hpf, overlay.FUNC_GRAIN_MERGE
            # )

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
                    ("fd_overlay_angle_var", fd_overlay_angle_var),
                    ("fd_overlay_hpf", fd_overlay_hpf),
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
    
        elif "edge_angle_variance" in result_dir:
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
                    "edge_angle_variance/angle_variance.tiff"
                ),
                cv2.IMREAD_UNCHANGED
            )
            fd_angle_var = (cm.get_cmap("jet")(
                fd_angle_var / fd_angle_var.max()
            ) * 255).astype(np.uint8)[:, :, [2, 1, 0]]
        
            # Generate Image of Confusion Matrix
            confusion_matrix = np.dstack(
                [result, ground_truth, result]
            )
        
            missing = confusion_matrix.copy()
            missing[~np.all(missing == C_GREEN, axis=2)] = [0, 0, 0]
        
            wrong = confusion_matrix.copy()
            wrong[~np.all(wrong == C_MAGENTA, axis=2)] = [0, 0, 0]
        
            # Overlay
            fd_overlay_angle_var = hsv_blending(
                src_gs,
                fd_angle_var
            )
            # fd_overlay_angle_var = overlay.do_gimp_overlay(
            #     src_gs, fd_angle_var, overlay.FUNC_GRAIN_MERGE
            # )
        
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
        
            write_images(
                result_dir,
                [
                    ("fd_overlay_angle_var", fd_overlay_angle_var),
                    ("confusion_matrix", confusion_matrix),
                    ("missing", missing),
                    ("wrong", wrong),
                    ("missing_extracted", missing_extracted),
                    ("missing_extracted_with_color", missing_extracted_with_color),
                    ("missing_extracted_by_anglevar", missing_extracted_by_anglevar),
                    ("wrong_extracted", wrong_extracted),
                    ("wrong_extracted_with_color", wrong_extracted_with_color),
                    ("wrong_extracted_by_anglevar", wrong_extracted_by_anglevar)
                ]
            )
    
    
        elif "edge_pixel_classify" in result_dir:
            result = (imread_with_error(
                join(
                    result_dir,
                    "building_damage.tiff"
                ),
                cv2.IMREAD_UNCHANGED
            ) * 255).astype(np.uint8)

            fd = imread_with_error(
                join(
                    result_dir,
                    "features.tiff"
                ),
                cv2.IMREAD_UNCHANGED
            )
            fd = (cm.get_cmap("jet")(
                fd / fd.max()
            ) * 255).astype(np.uint8)[:, :, [2, 1, 0]]
            
            # Overlay
            fd_overlay = hsv_blending(
                src_gs,
                fd
            )
            # fd_overlay = overlay.do_gimp_overlay(
            #     src_gs, fd, overlay.FUNC_GRAIN_MERGE
            # )

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
            missing_extracted_by_fd = merge_arrays_by_mask(
                (src_gs * CONT).astype(np.uint8),
                fd,
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
            wrong_extracted_by_fd = merge_arrays_by_mask(
                (src_gs * CONT).astype(np.uint8),
                fd,
                np.all(wrong == C_MAGENTA, axis=2)
            )

            write_images(
                result_dir,
                [
                    ("fd_overlay", fd_overlay),
                    ("confusion_matrix", confusion_matrix),
                    ("missing", missing),
                    ("wrong", wrong),
                    ("missing_extracted", missing_extracted),
                    ("missing_extracted_with_color", missing_extracted_with_color),
                    ("missing_extracted_by_fd", missing_extracted_by_fd),
                    ("wrong_extracted", wrong_extracted),
                    ("wrong_extracted_with_color", wrong_extracted_with_color),
                    ("wrong_extracted_by_fd", wrong_extracted_by_fd)
                ]
            )


def only_overlay_image():
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
        # for gt_type in ["GT_BOTH", "GT_ORANGE", "GT_RED"]
        for gt_type in ["GT_BOTH"]
    ], [])
    
    result_dirs = sorted(result_dirs)
    
    for result_dir in result_dirs:
        
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
        
        if "edge_angle_variance_with_hpf" in result_dir:
            fd_angle_var = imread_with_error(
                join(
                    result_dir,
                    "edge_angle_variance/angle_variance.tiff"
                ),
                cv2.IMREAD_UNCHANGED
            )
            fd_angle_var = (cm.get_cmap("jet")(
                fd_angle_var / fd_angle_var.max()
            ) * 255).astype(np.uint8)[:, :, [2, 1, 0]]
            
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
            
            # Overlay
            fd_overlay_angle_var = hsv_blending(
                src_gs,
                fd_angle_var
            )
            # fd_overlay_angle_var = overlay.do_gimp_overlay(
            #     src_gs, fd_angle_var, overlay.FUNC_GRAIN_MERGE
            # )
            fd_overlay_hpf = hsv_blending(
                src_gs,
                fd_hpf
            )
            # fd_overlay_hpf = overlay.do_gimp_overlay(
            #     src_gs, fd_hpf, overlay.FUNC_GRAIN_MERGE
            # )
            
            write_images(
                result_dir,
                [
                    ("fd_overlay_angle_var", fd_overlay_angle_var),
                    ("fd_overlay_hpf", fd_overlay_hpf)
                ]
            )
        
        elif "edge_angle_variance" in result_dir:
            fd_angle_var = imread_with_error(
                join(
                    result_dir,
                    "edge_angle_variance/angle_variance.tiff"
                ),
                cv2.IMREAD_UNCHANGED
            )
            fd_angle_var = (cm.get_cmap("jet")(
                fd_angle_var / fd_angle_var.max()
            ) * 255).astype(np.uint8)[:, :, [2, 1, 0]]
            
            # Overlay
            fd_overlay_angle_var = hsv_blending(
                src_gs,
                fd_angle_var
            )
            # fd_overlay_angle_var = overlay.do_gimp_overlay(
            #     src_gs, fd_angle_var, overlay.FUNC_GRAIN_MERGE
            # )
            
            write_images(
                result_dir,
                [
                    ("fd_overlay_angle_var", fd_overlay_angle_var)
                ]
            )
        
        
        elif "edge_pixel_classify" in result_dir:
            fd = imread_with_error(
                join(
                    result_dir,
                    "features.tiff"
                ),
                cv2.IMREAD_UNCHANGED
            )
            fd = (cm.get_cmap("jet")(
                fd / fd.max()
            ) * 255).astype(np.uint8)[:, :, [2, 1, 0]]
            
            # Overlay
            fd_overlay = hsv_blending(
                src_gs,
                fd
            )
            # fd_overlay = overlay.do_gimp_overlay(
            #     src_gs, fd, overlay.FUNC_GRAIN_MERGE
            # )
            
            write_images(
                result_dir,
                [
                    ("fd_overlay", fd_overlay)
                ]
            )


def eval_by_utilize_methods():
    ROOT_DIR_RESULT = "/Users/popunbom/Google Drive/情報学部/研究/修士/最終発表/Thesis/img/result"
    
    SPLIT_PATTERNS = {
        "MEAN_SHIFT"                                      : (0,),
        "ANGLE_VARIANCE"                                  : (1,),
        "PIXEL_CLASSIFY"                                  : (2,),
        "MEAN_SHIFT_AND_ANGLE_VARIANCE"                   : (0, 1),
        "MEAN_SHIFT_AND_PIXEL_CLASSIFY"                   : (0, 2),
        "ANGLE_VARIANCE_AND_PIXEL_CLASSIFY"               : (1, 2),
        "MEAN_SHIFT_AND_ANGLE_VARIANCE_AND_PIXEL_CLASSIFY": (0, 1, 2)
    }
    
    result_paths = sorted([
        join(dir_path, "result.png")
        for (dir_path, dir_names, file_names) in walk(ROOT_DIR_RESULT)
        if "result.png" in file_names
    ])
    
    for result_path in result_paths:
        exp_num, = re.match(
            r".*aerial_roi([0-9]).*",
            result_path
        ).groups()
        
        result_src = imread_with_error(
            result_path
        )
        
        gt_src = imread_with_error(
            join(
                ROOT_DIR_GT,
                f"aerial_roi{exp_num}.png"
            )
        )
        
        result_dir = join(
            dirname(result_path),
            "evaluation"
        )
        if not exists(result_dir):
            makedirs(result_dir)
        
        NDARRAY_ASSERT(result_src, ndim=3)
        
        print("File:", result_path)
        print(
            np.unique(
                result_src.reshape(result_src.shape[0] * result_src.shape[1], result_src.shape[2]),
                axis=0
            )
        )
        
        # Use Vegetation Mask
        OPTIONS = ["NORMAL"]
        
        path_vegetation_mask = join(
            ROOT_DIR_VEG_MASK,
            f"aerial_roi{exp_num}.png"
        )
        vegetation_mask = None
        
        if exists(path_vegetation_mask):
            vegetation_mask = imread_with_error(
                path_vegetation_mask,
                cv2.IMREAD_GRAYSCALE
            ).astype(bool)
            OPTIONS.append("REMOVED_VEGETATION")
            
        
        scores = dict()
        
        for option in OPTIONS:
        
            for name, channels in SPLIT_PATTERNS.items():
                eprint("Evaluation:", name)
                
                result = np.zeros(result_src.shape[:2], dtype=np.int16)
                
                for channel in channels:
                    result += (result_src[:, :, channel] != 0)

                result = result.astype(bool)

                if option == "NORMAL":
                    save_dir = result_dir
                elif option == "REMOVED_VEGETATION":
                    result = result & ~vegetation_mask
                    save_dir = join(
                        result_dir,
                        "removed_vegetation"
                    )
                
                if not exists(save_dir):
                    makedirs(save_dir)
    
                imwrite_with_error(
                    join(save_dir, name.replace(" & ", "_and_") + ".png"),
                    (result * 255).astype(np.uint8)
                )
                
                scores[name] = dict()
                
                for gt_type in ["GT_BOTH", "GT_ORANGE", "GT_RED"]:
                    ground_truth = None
                    
                    if gt_type == "GT_BOTH":
                        ground_truth = np.all(
                            (gt_src == C_RED) | (gt_src == C_ORANGE),
                            axis=2
                        )
                    elif gt_type == "GT_RED":
                        ground_truth = np.all(
                            gt_src == C_RED,
                            axis=2
                        )
                    elif gt_type == "GT_ORANGE":
                        ground_truth = np.all(
                            gt_src == C_ORANGE,
                            axis=2
                        )
                    
                    cm, metrics = evaluation_by_confusion_matrix(
                        result,
                        ground_truth
                    )
                    
                    scores[name][gt_type] = {
                        "Confusion Matrix": cm,
                        "Score"           : metrics
                    }
            
            
            json.dump(
                scores,
                open(join(save_dir, "scores.json"), "w"),
                ensure_ascii=False,
                sort_keys=True,
                indent="\t"
            )


def gen_result_road_damage():
    ROOT_DIR_RESULT = "/Users/popunbom/.tmp/EQDmgAnalyzr/detect_road_damage_v2"
    
    dirs = [
        join(ROOT_DIR_RESULT, d)
        for d in listdir(ROOT_DIR_RESULT)
        if isdir(join(ROOT_DIR_RESULT, d))
    ]
    
    for d in dirs:
        print(d)
        exp_num, = re.match(
            r".*aerial_roi([0-9]).*",
            d
        ).groups()
        
        src = imread_with_error(
            join(
                ROOT_DIR_SRC,
                f"aerial_roi{exp_num}.png"
            )
        )
        src_gs = cv2.cvtColor(
            cv2.cvtColor(
                src,
                cv2.COLOR_BGR2GRAY
            ),
            cv2.COLOR_GRAY2BGR
        )
        
        road_mask = imread_with_error(
            join(
                ROOT_DIR_ROAD_MASK,
                f"aerial_roi{exp_num}.png"
            ),
            cv2.IMREAD_GRAYSCALE
        ).astype(bool)
        
        result = imread_with_error(
            join(
                d,
                "result_extracted.tiff"
            ),
            cv2.IMREAD_UNCHANGED
        )
        
        src[road_mask == False] = [0, 0, 0]
        
        result = (cm.get_cmap("jet")(
            result / result.max()
        ) * 255).astype(np.uint8)[:, :, [2, 1, 0]]
        
        dst = hsv_blending(src, result)
        
        dst[road_mask == False] = src_gs[road_mask == False]
        
        imwrite_with_error(
            join(
                d,
                f"study_overlay.png"
            ),
            dst
        )


def eval_road_damage():
    ROOT_DIR_RESULT = "/Users/popunbom/Google Drive/情報学部/研究/修士/最終発表/Thesis/img/result"
    
    for exp_num in [1, 2, 3, 5, 9]:
        
        eprint(dedent(f"""
            Experiment Num: {exp_num}
        """))

        road_mask = imread_with_error(
            join(
                ROOT_DIR_ROAD_MASK,
                f"aerial_roi{exp_num}.png"
            ),
            cv2.IMREAD_GRAYSCALE
        ).astype(bool)

        # GT: GT_BOTH
        ground_truth = imread_with_error(
            join(
                ROOT_DIR_GT,
                f"aerial_roi{exp_num}.png"
            ),
            cv2.IMREAD_GRAYSCALE
        ).astype(bool)
        
        result_dir = join(
            ROOT_DIR_RESULT,
            f"aerial_roi{exp_num}/road_damage"
        )
        
        result_dirs = [result_dir]
        
        if exists(join(result_dir, "removed_vegetation")):
            result_dirs.append(
                join(result_dir, "removed_vegetation")
            )
            
        for result_dir in result_dirs:
            result = imread_with_error(
                join(result_dir, "thresholded.png"),
                cv2.IMREAD_GRAYSCALE
            ).astype(bool)
            
            result = result[road_mask == True]
            ground_truth_masked = ground_truth[road_mask == True]
            # ground_truth[road_mask == False] = False
            
            cm, metrics = evaluation_by_confusion_matrix(
                result,
                ground_truth_masked
            )
            
            result = {
                "Confusion Matrix": cm,
                "Score": metrics
            }
            
            json.dump(
                result,
                open(
                    join(result_dir, "scores.json"),
                    "w"
                ),
                ensure_ascii=False,
                sort_keys=True,
                indent="\t"
            )
    
    
if __name__ == '__main__':
    # gen_study_data()
    # only_overlay_image()
    # gen_result_road_damage()
    # eval_by_utilize_methods()
    eval_road_damage()
