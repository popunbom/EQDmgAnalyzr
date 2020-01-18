#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2019-11-28
# This is a part of EQDmgAnalyzr


import json
from os import path
import sys
from os.path import exists
from time import sleep

import cv2
import numpy as np

from damage_detector.building_damage import BuildingDamageExtractor
from utils.common import eprint
from utils.logger import ImageLogger


def test_all_procedures(path_src_img, path_ground_truth, parameters):
    """
    すべての手法をテストする
    
    Parameters
    ----------
    path_src_img : path-like object
        入力画像へのパス
    path_ground_truth : path-like object
        正解画像へのパス
    parameters : dict
        各処理における引数リスト

    """
    
    C_RED = [0, 0, 255]
    C_ORANGE = [0, 127, 255]
    
    IMG = cv2.imread(
        path_src_img,
        cv2.IMREAD_COLOR
    )
    
    GT = cv2.imread(
        path_ground_truth,
        cv2.IMREAD_COLOR
    )
    
    procedures = [
        "meanshift_and_color_thresholding",
        "edge_angle_variance_with_hpf",
        "edge_pixel_classify"
    ]
    
    for gt_opt in ["GT_BOTH", "GT_RED", "GT_ORANGE"]:
        if gt_opt == "GT_BOTH":
            ground_truth = np.all(
                (GT == C_RED) | (GT == C_ORANGE),
                axis=2
            )
        elif gt_opt == "GT_RED":
            ground_truth = np.all(
                GT == C_RED,
                axis=2
            )
        else:
            ground_truth = np.all(
                GT == C_ORANGE,
                axis=2
            )
        
        for proc_name in procedures:
            eprint("Do processing:", proc_name)
            logger = ImageLogger(
                "./tmp/detect_building_damage",
                prefix=path.splitext(
                    path.basename(
                        path_src_img
                    )
                )[0],
                suffix=proc_name + "_no_norm_" + gt_opt
            )
            inst = BuildingDamageExtractor(IMG, ground_truth, logger=logger)
            
            if proc_name in parameters:
                inst.__getattribute__(proc_name)(**parameters[proc_name])
            else:
                inst.__getattribute__(proc_name)()


def do_experiment(experiment_parameters):
    """
    実験ファイル (.json) に基づく実験を実行
    
    Parameters
    ----------
    experiment_parameters : dict
        実験ファイルデータ

    """
    
    
    def extract_parameters(parameters, case_num, procedure_name):
        """
        各手法におけるパラメータを抽出
        
        Parameters
        ----------
        parameters : list of dict
            パラメータリスト
        case_num : int
            実験番号
        procedure_name :str
            手法名称

        Returns
        -------
        dict
            抽出された引数リスト
        """
        
        parameters_by_case = [
            p
            for p in parameters
            if p["experiment_num"] == case_num
        ]
        
        if parameters_by_case:
            parameters_by_case = parameters_by_case[0]
            if procedure_name in parameters_by_case:
                return parameters_by_case[procedure_name]
            
        return dict()
    
    
    C_RED = [0, 0, 255]
    C_ORANGE = [0, 127, 255]
    
    p = experiment_parameters
    
    for case_num in p["options"]["experiments"]:
        # Check files existence
        img_paths = [
            path.join(
                p["resource_dirs"][img_type],
                "aerial_roi{n}.png".format(n=case_num)
            )
            for img_type in ["aerial_image", "ground_truth"]
        ]
        
        for img_path in img_paths:
            if not exists(img_path):
                raise FileNotFoundError(
                    "Not found image file -- {path}".format(
                        path=img_path
                    )
                )
        
        src_img, gt_img = [
            cv2.imread(img_path, cv2.IMREAD_COLOR)
            for img_path in img_paths
        ]
        
        for gt_opt in p["options"]["ground_truth"]:
            
            ground_truth = None
            
            if gt_opt == "GT_BOTH":
                ground_truth = np.all(
                    (gt_img == C_RED) | (gt_img == C_ORANGE),
                    axis=2
                )
            elif gt_opt == "GT_RED":
                ground_truth = np.all(
                    gt_img == C_RED,
                    axis=2
                )
            elif gt_opt == "GT_ORANGE":
                ground_truth = np.all(
                    gt_img == C_ORANGE,
                    axis=2
                )
                
            for procedure_name in p["options"]["procedures"]:
                eprint("Experiment Procedure:", procedure_name)
                
                logger = ImageLogger(
                    p["resource_dirs"]["logging"],
                    prefix="aerial_roi{n}".format(n=case_num),
                    suffix=procedure_name + "_no_norm_" + gt_opt
                )
                inst = BuildingDamageExtractor(src_img, ground_truth, logger=logger)
                
                inst.__getattribute__(procedure_name)(
                    **extract_parameters(p["parameters"], case_num, procedure_name)
                )
                
                # BEEP 5 TIMES (for notification)
                for _ in range(5):
                    sleep(1.0)
                    print("\a", end="", flush=True)
            
            for _ in range(5):
                sleep(1.0)
                print("\a", end="", flush=True)


if __name__ == '__main__':
    
    argc, argv = len(sys.argv), sys.argv
    
    if argc != 2:
        eprint(
            "usage: {prog} [experiment-json-file]".format(
                prog=argv[0]
            )
        )
        sys.exit(-1)
    
    else:
        try:
            do_experiment(json.load(open(argv[1])))
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            with open("error.log", "wt") as f:
                f.write(repr(e))
            # for DEBUG
            raise e
