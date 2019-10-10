# -*- coding: utf-8 -*-
# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2019-10-05

# This is a part of EQDmgAnalyzr

"""
imgproc/labeling.py : ラベリング処理
"""


import cv2
import numpy as np

from imgproc.utils import check_if_binary_image
from utils.assertion import *
from utils.common import eprint


def labeling_from_mask( img_bin ):
    """
    2値化画像へのラベリング
    
    - `cv2.connectedComponents` によるラベリング
    
    - ラベリング結果は各ラベルに属する座標値の配列
    
    Parameters
    ----------
    img_bin : numpy.ndarray
        入力画像 (2値化画像)

    Returns
    -------
    numpy.ndarray
        ラベリング結果

    """
    NDIM_ASSERT( img_bin, 2 )
    assert check_if_binary_image(img_bin), "'img_bin' must consist of binary values."
    
    
    if img_bin.max() != 255:
        img_bin = (img_bin == img_bin.max()).astype(np.uint8) * 255
    
    eprint( "Labeling... ", end="")

    n_labels, labels = cv2.connectedComponents( img_bin )

    eprint( "done! (Labels = {n_labels})".format(n_labels=n_labels) )
    
    eprint( "Create label array ... ", end="")
    
    labels_array = np.array([
        np.argwhere(labels == label_num)
        for label_num in range(n_labels)
    ])
    
    eprint( "done! " )
    
    return labels_array

