#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2020/01/28
# This is a part of EQDmgAnalyzr
import json
from collections import Counter
from os.path import join, exists
from textwrap import dedent

import pyperclip

from utils.common import eprint

TRANSLATE = {
    "METHODS": {
        "MEAN_SHIFT_AND_ANGLE_VARIANCE_AND_PIXEL_CLASSIFY": "色閾値処理＋エッジ角度分散＋エッジ画素分類",
        "MEAN_SHIFT_AND_ANGLE_VARIANCE"                   : "色閾値処理＋エッジ角度分散",
        "MEAN_SHIFT_AND_PIXEL_CLASSIFY"                   : "色閾値処理＋エッジ画素分類",
        "ANGLE_VARIANCE_AND_PIXEL_CLASSIFY"               : "エッジ角度分散＋エッジ画素分類",
        "MEAN_SHIFT"                                      : "色閾値処理のみ",
        "ANGLE_VARIANCE"                                  : "エッジ角度分散のみ",
        "PIXEL_CLASSIFY"                                  : "エッジ画素分類のみ",
    },
    "GT_TYPE": {
        "GT_BOTH"  : "全被害",
        "GT_ORANGE": "軽度被害のみ",
        "GT_RED"   : "重度被害のみ"
    },
    "HEADERS": {
        "Accuracy"    : "Accuracy",
        "Recall"      : "Recall",
        "Precision"   : "Precision",
        "Specificity" : "Specificity",
        "F Score"     : "F値",
        # "Missing-Rate": "未抽出率",
        # "Wrong-Rate"  : "誤抽出率",
    }
}

HEADERS = [
    "被害種別",
    "手法の組み合わせ"
]


def to_latex_table(exp_num):
    ROOT_DIR = "/Users/popunbom/Google Drive/情報学部/研究/修士/最終発表/Thesis/img/result"
    
    root_dir = join(
        ROOT_DIR, f"aerial_roi{exp_num}"
    )
    
    json_paths = list()
    
    json_paths.append(
        join(
            root_dir,
            "evaluation/scores.json"
        )
    )
    
    if exists(join(root_dir, "evaluation/removed_vegetation")):
        json_paths.append(
            join(
                root_dir,
                "evaluation/removed_vegetation/scores.json"
            )
        )
    
    codes = ""
    for json_path in json_paths:
        j = json.load(open(json_path))
        
        vegetation_status = ""
        if "removed_vegetation" in json_path:
            vegetation_status = ", 植生除去あり"
        
        caption = f"精度評価 (実験{exp_num}{vegetation_status})"
        
        records = list()
        
        # Insert Header
        records.append([
            *HEADERS,
            *TRANSLATE["HEADERS"].values()
        ])
        
        # Insert Each Record
        for k_gt_type, v_gt_type in TRANSLATE["GT_TYPE"].items():
            for k_method, v_method in TRANSLATE["METHODS"].items():
                records.append([
                    v_gt_type,
                    v_method,
                    *[
                        j[k_method][k_gt_type]["Score"][k]
                        for k in TRANSLATE["HEADERS"].keys()
                    ]
                ])
                
        # Row-cell merging
        
        first_row = [ records[i][0] for i in range(1, len(records)) ]
        
        c = Counter(first_row)
        
        prev = None
        for i, r in enumerate(first_row, start=1):
            if prev is None:
                prev = r
                records[i][0] = "\\multirow{{{n}}}{{*}}{{{value}}}".format(
                    n=c[r], value=r
                )
            else:
                if r == prev:
                    records[i][0] = ""
                else:
                    prev = r
                    records[i][0] = "\\multirow{{{n}}}{{*}}{{{value}}}".format(
                        n=c[r], value=r
                    )
        
        
        code = dedent("""
            \\begin{{table}}[H]
              \\begin{{center}}
                \\caption{{{caption}}}
                \\label{{tab:{caption}}}
                \\resizebox{{\\textwidth}}{{!}}{{
                  \\begin{{tabular}}{{clrrrrrrr}} \\hline
            
                    \\hline
                    {headers} \\\\
                    \\hline
                    
                    \\hline
{records} \\\\
                    \\hline
                    
                    \\hline
                  \\end{{tabular}}
                }}
              \\end{{center}}
            \\end{{table}}
        """.format(
            caption=caption,
            headers=" & ".join([
                f"\\textbf{{{h}}}"
                for h in records[0]
            ]),
            records=" \\\\\n".join([
                " " * 26 + " & ".join([
                    f"{x:.1f}" if isinstance(x, float) else x
                    for x in row
                ])
                for row in records[1:]
            ])
        ))
        
        codes += code + "\n"
        
    print(codes)
    pyperclip.copy(codes)
    eprint("Copied to clipboard")


if __name__ == '__main__':
    while True:
        exp_num = input("EXP NUM ? >")
        
        if exp_num:
            to_latex_table(int(exp_num))
