#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2020/02/16
# This is a part of EQDmgAnalyzr
from multiprocessing import current_process
from os.path import join
from pprint import pprint

import numpy as np
from pymeanshift import segment
from tqdm import tqdm

from imgproc.utils import imread_with_error
from utils.pool import CustomPool


ROOT_DIR_SRC = "./img/resource/aerial_image/CLAHE_with_WB_adjust"
ROOT_DIR_ANS = join(ROOT_DIR_SRC, "smoothed")

SP_RANGE = (1, 16, 1)
SR_RANGE = (1, 16, 1)


def func_worker(img, spatial_radius=None, range_radius=None, min_density=None):
    _worker_id = current_process()._identity[0]
    _desc = f"Worker #{_worker_id:3d} (sp={spatial_radius:2.1f}, sr={range_radius:2.1f})"
    
    for _ in tqdm([0], desc=_desc, position=_worker_id, leave=False):
        
        segmented = segment(
            img,
            spatial_radius=spatial_radius,
            range_radius=range_radius,
            min_density=min_density
        )[0]
    
    return segmented, spatial_radius, range_radius


def find_ms_params(n):
    file_name = f"aerial_roi{n}.png"
    
    src = imread_with_error(
        join(ROOT_DIR_SRC, file_name)
    )
    ans = imread_with_error(
        join(ROOT_DIR_ANS, file_name)
    )
    
    
    ms_params = sum([
        [
            {
                "spatial_radius": sp,
                "range_radius": sr,
                "min_density": 0
            }
            for sr in np.arange(SR_RANGE[0], SR_RANGE[0]+SR_RANGE[1], SR_RANGE[2])
        ]
        for sp in np.arange(SP_RANGE[0], SP_RANGE[0]+SP_RANGE[1], SP_RANGE[2])
    ], [])
    
    progress_bar = tqdm(total=len(ms_params), position=0)
    
    def _update_progressbar(arg):
        progress_bar.update()
    
    
    cp = CustomPool()
    pool = cp.Pool(n_process=6, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
    
    results = list()
    for params in ms_params:
        
        results.append(
            pool.apply_async(
                func_worker,
                args=(src, ),
                kwds=params,
                callback=_update_progressbar
            )
        )
    pool.close()
    pool.join()
    cp.update()
    
    results = [result.get() for result in results]
    
    results = sorted(
        [
            (
                sp,
                sr,
                np.sum(
                    np.abs(segmented - ans)
                )
            )
            for segmented, sp, sr in results
        ],
        key=lambda e: e[0]
    )
    
    pprint(results)
    
    with open(f"tmp/find_ms_params_{n}.csv", "wt") as f:
        f.write("spatial_radius, range_radius, n_diffs\n")
        for result in results:
            f.write(", ".join([ str(x) for x in result]) + "\n")
    
    return results


if __name__ == '__main__':
    for n in [1, 2, 3, 5, 9]:
        find_ms_params(n)
        
