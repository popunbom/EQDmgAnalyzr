# /usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: popunbom <fantom0779@gmail.com>
# Created At: 2019/08/29


import cv2
import numpy as np
from PIL import Image

from utils.assertion import TYPE_ASSERT
from utils.exception import UnsupportedDataType

from utils.common import check_module_avaliable

MAMBA_AVAILABLE = check_module_avaliable( "mamba" )

def pil2np( pil_img ):
    """
    PIL.Image.Image を numpy.ndarray に変換する
        REF: https://qiita.com/derodero24/items/f22c22b22451609908ee

    Parameters
    ----------
    pil_img : PIL.Image.Image
        変換元の PIL.Image.Image 画像データ

    Returns
    -------
    numpy.ndarray
        変換された numpy.ndarray 画像データ
    """
    TYPE_ASSERT( pil_img, Image.Image )
    
    npy_img = np.array( pil_img )
    
    # Data depth conversion
    if npy_img.dtype not in [np.uint8, np.float32]:
        npy_img = npy_img.astype( np.float32 )
        npy_img /= npy_img.max()
    
    if npy_img.ndim == 3:
        if npy_img.shape[2] == 3:
            # RGB -> BGR
            npy_img = cv2.cvtColor( npy_img, cv2.COLOR_RGB2BGR )
        elif npy_img.shape[2] == 4:
            # RGBA -> BGRA
            npy_img = cv2.cvtColor( npy_img, cv2.COLOR_RGBA2BGRA )
        else:
            raise UnsupportedDataType( "npy_img.shape = {shape}".format(
                shape=npy_img.shape
            ) )
    
    return npy_img


if MAMBA_AVAILABLE:
    import mamba as mb
    
    
    def np2mamba( npy_img ):
        """
        numpy.ndarray を mamba.base.imageMb に変換する
    
        Parameters
        ----------
        npy_img : numpy.ndarray
            変換元の numpy.ndarray 画像データ
    
        Returns
        -------
        mamba.base.imageMb
            変換された mamba.base.imageMb 画像データ
        """
        
        TYPE_ASSERT( npy_img, np.ndarray )
        
        if npy_img.dtype == np.bool:
            bit_depth = 1
        elif npy_img.dtype == np.uint8:
            bit_depth = 8
        elif npy_img.dtype == np.float32:
            bit_depth = 32
        else:
            raise UnsupportedDataType( "npy_img.dtype = {dtype}".format(
                dtype=npy_img.dtype
            ) )
        
        mb_img = mb.imageMb( npy_img.shape[1], npy_img.shape[0], bit_depth )
        
        if npy_img.ndim == 3:
            npy_img = cv2.cvtColor( npy_img, cv2.COLOR_BGR2RGB )
        
        mb.PIL2Mamba( Image.fromarray( npy_img ), mb_img )
        
        return mb_img
    
    
    # mamba.base.imageMb --> numpy.ndarray
    def mamba2np( mb_img ):
        """
        mamba.base.imageMb を numpy.ndarray に変換する
    
        Parameters
        ----------
        mb_img : mamba.base.imageMb
            変換元の mamba.base.imageMb 画像データ
    
        Returns
        -------
        numpy.ndarray
            変換された numpy.ndarray 画像データ
        """
        TYPE_ASSERT( mb_img, mb.imageMb )
        
        return pil2np( mb.Mamba2PIL( mb_img ) )
