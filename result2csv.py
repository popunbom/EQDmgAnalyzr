#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2020-01-13
# This is a part of EQDmgAnalyzr


import json
import re
import sys
from os import listdir, path
from os.path import isdir
from pathlib import Path
from textwrap import dedent
import pyperclip

from utils.common import eprint

RESULT_ROOT_DIR = "./tmp/detect_building_damage/master-v2/fixed_histogram/GT_ORANGE"

EXP_PREFIX = "aerial_roi"

dirs = [d for d in listdir(RESULT_ROOT_DIR) if isdir(path.join(RESULT_ROOT_DIR, d))]

def get_dir_name(exp_num, exp_name):
    P = r"aerial_roi{exp_num}_[0-9]{{8}}_[0-9]{{6}}_{exp_name}".format(
        exp_num=exp_num,
        exp_name=exp_name
    )
    return [d for d in dirs if re.match(P, d) is not None][0]

def gen_result(exp_num, exp_name):
    result = ""
    
    dir_path = Path(path.join(
        RESULT_ROOT_DIR,
        get_dir_name(exp_num, exp_name)
    ))
    
    if exp_name == "edge_angle_variance_with_hpf":
        d = json.load(open(dir_path / "edge_angle_variance/params.json"))
        
        result += dedent(f"""
            {d['window_proc']['window_size']}
            {d['window_proc']['step']}
        """).lstrip('\n')
        
        d = json.load(open(dir_path / "high_pass_filter/params.json"))
        
        result += dedent(f"""
            {d['freq']}
            {d['window_proc']['window_size']}
            {d['window_proc']['step']}
        """).lstrip('\n')
        
        d = json.load(open(dir_path / "params_finder/params_subtracted_thresholds.json"))
        
        result += dedent(f"""
            [{d['Range']['img_a'][0]:.4f}, {d['Range']['img_a'][1]:.4f}]
            [{d['Range']['img_b'][0]:.4f}, {d['Range']['img_b'][1]:.4f}]
            {d['Score']['Accuracy']:.04f}
            {d['Score']['Precision']:.04f}
            {d['Score']['Recall']:.04f}
            {d['Score']['Specificity']:.04f}
            {d['Score']['Missing-Rate']:.04f}
            {d['Score']['Wrong-Rate']:.04f}
            {d['Score']['F Score']:.04f}
        """).lstrip('\n')
    
    elif exp_name == "edge_angle_variance":
        d = json.load(open(dir_path / "edge_angle_variance/params.json"))
        
        result += dedent(f"""
            {d['window_proc']['window_size']}
            {d['window_proc']['step']}
        """).lstrip('\n')
        
        d = json.load(open(dir_path / "params_finder_angle_variance/params.json"))
        
        result += dedent(f"""
            [{d['Range'][0]:.4f}, {d['Range'][1]:.4f}]
            {d['Score']['Accuracy']:.04f}
            {d['Score']['Precision']:.04f}
            {d['Score']['Recall']:.04f}
            {d['Score']['Specificity']:.04f}
            {d['Score']['Missing-Rate']:.04f}
            {d['Score']['Wrong-Rate']:.04f}
            {d['Score']['F Score']:.04f}
        """).lstrip('\n')
    
    elif exp_name == "edge_pixel_classify":
        d = json.load(open(dir_path / "params.json"))
    
        result += dedent(f"""
            {d['canny']['sigma']}
            {d['canny']['low_threshold']}
            {d['canny']['high_threshold']}
            {d['window_proc']['window_size']}
            {d['window_proc']['step']}
        """).lstrip('\n')
        
        d = json.load(open(dir_path / "params_finder_edge_line_feature/params.json"))
    
        result += dedent(f"""
            [{d['Range'][0]:.4f}, {d['Range'][1]:.4f}]
            {d['Score']['Accuracy']:.04f}
            {d['Score']['Precision']:.04f}
            {d['Score']['Recall']:.04f}
            {d['Score']['Specificity']:.04f}
            {d['Score']['Missing-Rate']:.04f}
            {d['Score']['Wrong-Rate']:.04f}
            {d['Score']['F Score']:.04f}
        """).lstrip('\n')
    
    elif exp_name == "meanshift_and_color_thresholding":
        d = json.load(open(dir_path / "detail_mean_shift.json"))
    
        result += dedent(f"""
            {d['Mean-Shift']['params']['spatial_radius']}
            {d['Mean-Shift']['params']['range_radius']}
        """).lstrip('\n')
        
        d = json.load(open(dir_path / "params_finder/color_thresholds_in_hsv.json"))
    
        result += dedent(f"""
            [{d['Range']['H'][0]:.0f}, {d['Range']['H'][1]:.0f}]
            [{d['Range']['S'][0]:.0f}, {d['Range']['S'][1]:.0f}]
            [{d['Range']['V'][0]:.0f}, {d['Range']['V'][1]:.0f}]
        """).lstrip('\n')
        
        d = json.load(open(dir_path / "params_finder/params_morphology.json"))
        
        s = json.load(open(dir_path / "scores_fixed_result.json"))
        
        m = {
            "ERODE": "収縮",
            "DILATE": "膨張",
            "OPEN": "オープニング",
            "CLOSE": "クロージング"
        }
    
        result += dedent(f"""
            {d['Params']['Kernel']['Size'][0]}px × {d['Params']['Kernel']['Size'][1]}px
            {d['Params']['Kernel']['#_of_Neighbor']}近傍
            {m[d['Params']['Operation']]}
            {d['Params']['Iterations']}回
            {s['Score']['Accuracy']:.04f}
            {s['Score']['Precision']:.04f}
            {s['Score']['Recall']:.04f}
            {s['Score']['Specificity']:.04f}
            {s['Score']['Missing-Rate']:.04f}
            {s['Score']['Wrong-Rate']:.04f}
            {s['Score']['F Score']:.04f}
        """).lstrip('\n')
        

    return result


if __name__ == '__main__':
    EXP_NAMES = [
        "edge_angle_variance_with_hpf",
        "edge_angle_variance",
        "edge_pixel_classify",
        "meanshift_and_color_thresholding"
    ]
    
    argc, argv = len(sys.argv), sys.argv
    
    if argc == 2:
        exp_num = int(argv[1])
        results = "\n".join([gen_result(exp_num, exp_name) for exp_name in EXP_NAMES])
        print(results)
        pyperclip.copy(results)
        eprint("Copied to clipboard")

    else:
        while True:
            line = input(dedent(f"""
            ROOT_PATH: {RESULT_ROOT_DIR}
            EXP NUM? >
            """).rstrip("\n"))
            if line:
                exp_num = int(line)
                results = "\n".join([gen_result(exp_num, exp_name) for exp_name in EXP_NAMES])
                print(results)
                pyperclip.copy(results)
                eprint("Copied to clipboard")
            else:
                break
