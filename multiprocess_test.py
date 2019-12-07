# /usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: popunbom <fantom0779@gmail.com>
# Created At: 2019/12/06
import functools
import re

import cv2
import numpy as np

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


if __name__ == '__main__':
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
    ).astype(bool)
    
    logger = ImageLogger(
        base_path="./tmp/multiprocess_test"
    )

    thresholds = range(256)
    
    with Pool(processes=4, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as p:
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
                total=len(thresholds)
            )
        ]
    
    reasonable_params, result = max(
        results,
        key=lambda result: result[0]["Score"]
    )
    
    if result.dtype != bool:
        result = (result * 255).astype(np.uint8)
    
    logger.logging_dict(reasonable_params, "canny_thresholds")
    logger.logging_img(result, "canny")
