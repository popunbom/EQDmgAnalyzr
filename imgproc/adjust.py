#/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: popunbom <fantom0779@gmail.com>
# Created At: 2019/08/29
import cv2

from utils.assertion import NDIM_ASSERT


def equalizeHist( img, mode='hsv' ):
    """
    ヒストグラム平坦化を行う

    Parameters
    ----------
    img : numpy.ndarray
        入力画像データ。画像はグレースケールまたは RGB カラー画像である必要がある。
    mode : str
        動作切り替え用のパラメータ。デフォルト値は 'hsv'。
        mode='hsv' のとき、HSV色空間の各チャンネルに対してヒストグラム平坦化を行う
        mode='rgb' のとき、RGB色空間の各チャンネルに対してヒストグラム平坦化を行う

    Returns
    -------
    img : numpy.ndarray
        ヒストグラム平坦化が施された画像データ。入力画像と同じデータ型、チャンネル数の
        画像が返却される。

    """
    NDIM_ASSERT(img, (2, 3))
    assert mode in ['hsv', 'rgb'], \
        f"""'mode' must be '{"' or '".join( ['hsv', 'rgb'] )}'"""
    
    if img.ndim == 2:
        return cv2.equalizeHist( img )
    
    else:
        if mode == 'hsv':
            img = cv2.cvtColor( img, cv2.COLOR_BGR2HSV )
            img[:, :, 2] = cv2.equalizeHist( img[:, :, 2] )
            img = cv2.cvtColor( img, cv2.COLOR_HSV2BGR )
        
        elif mode == 'rgb':
            for i in range( img.ndim ):
                img[:, :, i] = cv2.equalizeHist( img[:, :, i] )
        
        return img
