#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2020/01/25
# This is a part of EQDmgAnalyzr

import cv2
import numpy as np

from utils.assertion import NDARRAY_ASSERT, SAME_SHAPE_ASSERT
from utils.exception import UnsupportedOption
from utils.reflection import get_var_name


def _check_arrays(I, M):
    NDARRAY_ASSERT(I, dtype=np.uint8)
    NDARRAY_ASSERT(M, dtype=np.uint8)
    
    SAME_SHAPE_ASSERT(I, M, ignore_ndim=True)


def _expand_channels(I, M):
    ndim_i, ndim_m = I.ndim, M.ndim
    
    if ndim_i > ndim_m:
        I = cv2.cvtColor(I.copy(), cv2.COLOR_GRAY2BGR)
    elif ndim_i < ndim_m:
        M = cv2.cvtColor(M.copy(), cv2.COLOR_GRAY2BGR)
    
    return I, M


def _grain_merge(I, M):
    I, M = _expand_channels(I, M)
    
    I = I.astype(np.int16)
    M = M.astype(np.int16)
    
    return (I + M - 128).astype(np.uint8)


FUNC_GRAIN_MERGE = 1


MAP_FUNCION = {
    FUNC_GRAIN_MERGE: _grain_merge
}


def do_gimp_overlay(I, M, func):
    _check_arrays(I, M)
    
    if func not in MAP_FUNCION.keys():
        raise UnsupportedOption(
            "Not implemented Options: {}".format(
                get_var_name(func)
            ),
            available_options=[
                get_var_name(k)
                for k in MAP_FUNCION.keys()
            ]
        )
    
    return MAP_FUNCION[func](I, M)
