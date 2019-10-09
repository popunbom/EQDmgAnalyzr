# -*- coding: utf-8 -*-
# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2019-10-05

# This is a part of EQDmgAnalyzr

"""
imgproc/preproc.py : 画像の前処理に関わるもの
"""

import cv2

from utils.assertion import *


def denoise_by_median_and_meanshift( src_img ):
    """
    メディアンフィルタと Mean-Shift による平滑化処理

    1. RGB -> HSV への変換

    2. 各平滑化処理

       - H, S にメディアンフィルタ (`r=5`)

       - V に Mean-Shift (`spacial=8, chroma=18`)

    3. HSV -> RGB 逆変換

    Parameters
    ----------
    src_img : numpy.ndarray
        入力画像 (RGB カラー)

    Returns
    numpy.ndarray
        平滑化された画像
    -------

    """
    NDIM_ASSERT( src_img, 3 )
    
    print( "Image Pre-Processing ... ", flush=True, end="" )
    
    hsv = cv2.cvtColor( src_img, cv2.COLOR_BGR2HSV )
    h, s, v = [hsv[:, :, ch] for ch in range( 3 )]
    
    # Hue, Saturation: Median( r = 5.0 )
    h = cv2.medianBlur( h, ksize=5 )
    s = cv2.medianBlur( s, ksize=5 )
    
    # Value: MeanShift ( spacial=8 chroma=18 )
    v = cv2.cvtColor( v, cv2.COLOR_GRAY2BGR )
    v = cv2.pyrMeanShiftFiltering( v, sp=8, sr=18 )
    v = cv2.cvtColor( v, cv2.COLOR_BGR2GRAY )
    
    dst = cv2.cvtColor( np.dstack( (h, s, v) ), cv2.COLOR_HSV2BGR )
    
    print( "done! ", flush=True )
    
    return dst

def equalized_histogram_using_mask( img, mask, mode="rgb" ):
    """
    マスク画像を考慮したヒストグラム平滑化
    
    - マスク が `True, 1, 255` の値の部分のみを考慮して
      ヒストグラム平滑化を行う
    
    Parameters
    ----------
    img : numpy.ndarray
        入力画像 (RGB カラー画像)
    mask : numpy.ndarray
        マスク画像 (8-Bit グレースケール画像)
    mode : str
        モード切り替え
        
        - "rgb" を指定した場合、RGB 色空間での
          ヒストグラム平滑化を行う
        
        - "hsv" を指定した場合、HSV 色空間での
          ヒストグラム平滑化を行う

    Returns
    -------
    numpy.ndarray
        ヒストグラム平滑化された画像
    """
    
    NDIM_ASSERT( img, 3 )
    NDIM_ASSERT( mask, 2 )
    TYPE_ASSERT(mode, str)
    assert mode in ("rgb", "hsv"), "'mode' must be 'rgb' or 'hsv'"
    
    m = (mask / mask.max()).flatten()
    
    # Make Mask
    mask = mask.astype( np.int16 )
    mask[mask > 0] = 1
    mask = mask.astype( np.uint8 )
    
    if mode == "hsv":
        img = cv2.cvtColor( img, cv2.COLOR_BGR2HSV )
        h, s, v = [ img[:, :, ch] for ch in range(3) ]
    
        for i, ch in zip( (1, 2), [s, v] ):
            ch = ch.flatten()
            x_masked = ch[m == 1]
            hist, bins = np.histogram( x_masked, 256, [0, 255] )
            
            cdf = hist.cumsum()
            
            cdf_m = np.ma.masked_equal( cdf, 0 )
            cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
            
            cdf = np.ma.filled( cdf_m, 0 ).astype( 'uint8' )
            
            img[:, :, i] = cdf[ch].reshape( img.shape[:2] )
        
        equalized = cv2.cvtColor( img, cv2.COLOR_HSV2BGR )
    
        for i in range( 3 ):
            equalized[:, :, i] = equalized[:, :, i] * mask
            
    elif mode == "rgb":
        b, g, r = [ img[:, :, ch] for ch in range(3) ]
        equalized = np.zeros( img.shape, dtype=np.uint8 )
    
        for i, x in enumerate( [b, g, r] ):
            x = x.flatten()
            x_masked = x[m == 1]
            hist, bins = np.histogram( x_masked, 256, [0, 256] )
            cdf = hist.cumsum()
        
            cdf_m = np.ma.masked_equal( cdf, 0 )
            cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        
            cdf = np.ma.filled( cdf_m, 0 ).astype( 'uint8' )
        
            equalized[:, :, i] = cdf[x].reshape( img.shape[:2] ) + mask
        
    
    return equalized
