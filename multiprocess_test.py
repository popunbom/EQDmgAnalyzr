# /usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: popunbom <fantom0779@gmail.com>
# Created At: 2019/12/06
import functools
import re
import time
from random import randint, random

import cv2
import numpy as np

from imgproc.utils import mp_compute_by_window
from utils.evaluation import evaluation_by_confusion_matrix

from multiprocessing import Pool, current_process
from tqdm import trange, tqdm

from utils.logger import ImageLogger

def mp_find_canny_thresholds(th_1, img, ground_truth):
    from skimage.feature import canny
    
    reasonable_params = {
        "Score"           : -1,
        "Confusion Matrix": None,
        "Thresholds"      : None,
    }
    result = None
    
    desc = f"In Range: ({th_1:3d}, 255] "
    worker_id = int(re.match(r"(.*)-([0-9]+)$", current_process().name).group(2))

    
    for th_2 in trange(th_1 + 1, 256, desc=desc, position=worker_id, leave=False):
        _result = canny(img, low_threshold=th_1, high_threshold=th_2)
        
        cm, metrics = evaluation_by_confusion_matrix(_result, ground_truth)
        
        if metrics["F Score"] > reasonable_params["Score"]:
            reasonable_params = {
                "Score"           : metrics["F Score"],
                "Confusion Matrix": cm,
                "Thresholds"      : list([th_1, th_2])
            }
            result = _result.copy()
    
    return reasonable_params, result

def test_1():
    # PATH_SRC_IMG = "img/resource/aerial_roi1_raw_ms_40_50.png"
    PATH_SRC_IMG = "img/resource/aerial_roi1_raw_denoised_clipped.png"
    # PATH_SRC_IMG = "img/resource/aerial_roi2_raw.png"
    
    PATH_GT_IMG = "img/resource/ground_truth/aerial_roi1.png"
    # PATH_GT_IMG = "img/resource/ground_truth/aerial_roi2.png"
    
    img = cv2.imread(
        PATH_SRC_IMG,
        cv2.IMREAD_COLOR
    )
    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )
    
    ground_truth = cv2.imread(
        PATH_GT_IMG,
        cv2.IMREAD_GRAYSCALE
    ).astype( bool )
    
    logger = ImageLogger(
        base_path="./tmp/multiprocess_test"
    )
    
    thresholds = range( 256 )
    
    with Pool( processes=4, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),) ) as p:
        results = [
            result for result in tqdm(
                p.imap_unordered(
                    func=functools.partial(
                        mp_find_canny_thresholds,
                        img=img,
                        ground_truth=ground_truth
                    ),
                    iterable=thresholds
                ),
                total=len( thresholds )
            )
        ]
    
    reasonable_params, result = max(
        results,
        key=lambda result: result[0]["Score"]
    )
    
    if result.dtype != bool:
        result = (result * 255).astype( np.uint8 )
    
    logger.logging_dict( reasonable_params, "canny_thresholds" )
    logger.logging_img( result, "canny" )

def mp_stddev(img):
    return np.std(img)

def test_2():
    # PATH_SRC_IMG = "img/resource/aerial_roi1_raw_ms_40_50.png"
    PATH_SRC_IMG = "img/resource/aerial_roi1_raw_denoised_clipped.png"
    # PATH_SRC_IMG = "img/resource/aerial_roi2_raw.png"
    
    PATH_GT_IMG = "img/resource/ground_truth/aerial_roi1.png"
    # PATH_GT_IMG = "img/resource/ground_truth/aerial_roi2.png"
    
    img = cv2.imread(
        PATH_SRC_IMG,
        cv2.IMREAD_COLOR
    )
    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )
    
    ground_truth = cv2.imread(
        PATH_GT_IMG,
        cv2.IMREAD_GRAYSCALE
    ).astype( bool )
    
    logger = ImageLogger(
        base_path="./tmp/multiprocess_test"
    )

    fd = mp_compute_by_window(
        img,
        func=mp_stddev,
        window_size=8,
        step=2
    )
    
    logger.logging_img("fd", fd)

def test_3(img, ws, func):
    
    global _func
    _func = func
    
    global _callee
    def _callee(_roi, _ws, _func):
        
        _worker_id = int( re.match( r"(.*)-([0-9]+)$", current_process().name ).group( 2 ) )
        _desc = f"Worker #{_worker_id}"

        _results = [
            _func(_roi[:, j:j+ws])
            for j in trange( 0, _roi.shape[1], _ws, desc=_desc, position=_worker_id, leave=False )
        ]

        return _results
        
    
    rois = [ img[i:i+ws, :] for i in range(0, img.shape[0], ws) ]
    
    pbar = tqdm(total=len(rois))

    def _update_progressbar(*args):
        pbar.update()
        

    p = Pool( processes=4, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),) )
    
    results = list()
    
    for roi in rois:
        results.append(
            p.apply_async(
                _callee,
                args=(roi, ws, func),
                callback=_update_progressbar
         )
    )
    p.close()
    p.join()
    
    return [ result.get() for result in results ]


def calc_f(img):
    time.sleep(0.01)
    return np.std(img)

if __name__ == '__main__':
    # test_1()
    # test_2()
    img = cv2.imread(
        "img/resource/aerial_roi1_raw_denoised_clipped.png",
        cv2.IMREAD_GRAYSCALE
    )
    
    test_3(
        img,
        8,
        func=calc_f
    )
