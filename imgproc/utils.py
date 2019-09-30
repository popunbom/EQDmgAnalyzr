# /usr/bin/env python3
# -*- coding: utf-8 -*-

# SharedProcedures : 各モジュールで共通する処理
# Author: popunbom <fantom0779@gmail.com>
# Created At: 2019/06/21
import os

import cv2
import numpy as np

from math import ceil

from utils.assertion import TYPE_ASSERT, SAME_SHAPE_ASSERT
from utils.common import n_args, eprint


def compute_by_window( imgs, func, window_size=16, step=2, dst_dtype=np.float32 ):
    """
    画像を一部を切り取り、func 関数で行った計算結果を
    返却する

    Parameters
    ----------
    imgs : numpy.ndarray or tuple of numpy.ndarray
        入力画像
        tuple で複数画像を与える場合、各画像に対して
        同じ領域を切り取り、処理を行うため、各画像の
        縦、横サイズは一致している必要がある
    func : callable object
        切り取った画像の一部に対して何らかの計算を行う
        関数。引数として画像の一部が渡される。
    window_size : int or tuple of int
        画像を切り取るサイズ。
        int を指定した場合は、縦横同じサイズで切り取る。
        tuple(int, int) を指定した場合は、縦横で異なったサイズ
        で切り取り、指定する順序は ndarray の次元に対応する
    step : int or tuple of int
        切り取り間隔
        int を指定した場合は、縦横同じ間隔を開けて処理をする
        tuple(int, int) を指定した場合は、縦横で異なった間隔
        を開けて処理を行い、指定する順序は ndarray の次元に
        対応する

    Returns
    -------
    numpy.ndarray
        各切り取り画像に対する処理結果の行列
    """
    TYPE_ASSERT( imgs, [np.ndarray, tuple] )

    if isinstance( imgs, tuple ):
        for img in imgs:
            TYPE_ASSERT( img, np.ndarray )
        for i in range( len( imgs ) - 1 ):
            SAME_SHAPE_ASSERT( imgs[i], imgs[i + 1] )
        n_imgs = len( imgs )
        height, width = imgs[0].shape[:2]
    else:
        n_imgs = 1
        height, width = imgs.shape[:2]

    # assert callable( func ) and func.__code__.co_argcount >= n_imgs, \
    assert callable( func ) and n_args( func ) >= n_imgs, \
        "argument 'func' must be callable object which has {0} argument at least. \n".format( n_imgs ) + \
        "  ( num of argumets of 'func' depends on argument 'imgs')"

    TYPE_ASSERT( window_size, [int, tuple] )
    TYPE_ASSERT( step, [int, tuple] )

    if isinstance( step, int ):
        s_i, s_j = [step] * 2
    else:
        s_i, s_j = step

    if isinstance( window_size, int ):
        w_i, w_j = [window_size] * 2
    else:
        w_i, w_j = window_size

    results = np.ndarray(
        (
            ceil( height / s_i ),
            ceil( width / s_j )
        ),
        dtype=dst_dtype
    )

    for ii, i in enumerate( range( 0, height, s_i ) ):
    
        for jj, j in enumerate( range( 0, width, s_j ) ):
        
            eprint( "\rWindow calculating ... {1:{0}d} / {2:{0}d}".format(
                len( str( results.size ) ),
                (jj + 1) + ii * results.shape[1],
                results.size
            ), end="" )
        
            if isinstance( imgs, np.ndarray ):
                roi = imgs[i: i + w_i, j: j + w_j]
                results[ii][jj] = func( roi )
            
            else:
                rois = [
                    img[i: i + w_i, j: j + w_j]
                    for img in imgs
                ]
    
                results[ii][jj] = func( *rois )

    return results


def get_rect( img_shape, points ):
    yMax, xMax = 0, 0
    yMin, xMin = img_shape
    
    points = np.array( points, dtype=np.int32 )
    
    yMin = min( np.min( points[:, 0] ), yMin )
    xMin = min( np.min( points[:, 1] ), xMin )
    yMax = max( np.max( points[:, 0] ), yMax )
    xMax = max( np.max( points[:, 1] ), xMax )
    
    if yMax <= yMin or xMax <= xMin:
        raise ValueError( "Value Range Error: yMax <= yMin || xMax <= xMin" )
    
    # return [yMin, yMax, xMin, xMax]
    return [(yMin, xMin), (yMax, xMax)]


def divide_by_mask( src_img, npy_label, dir_name ):
    assert (src_img.shape[2] == 3), \
        " 'src_img' must be 3-ch RGB image."
    
    if (not os.path.exists( "img/divided/" + dir_name )):
        os.mkdir( "img/divided/" + dir_name )
    
    for i in range( len( npy_label ) ):
        (yMin, xMin), (yMax, xMax) = get_rect( src_img.shape, npy_label[i] )
        
        cv2.imwrite(
            "img/divided/{dir_name}/{file_name:05d}.png".format(
                dir_name=dir_name,
                file_name=i
            ),
            src_img[yMin:yMax, xMin:xMax]
        )
        cv2.imwrite(
            "img/divided/{dir_name}/canny_{file_name:05d}.png".format(
                dir_name=dir_name,
                file_name=i
            ),
            cv2.Canny( src_img[yMin:yMax, xMin:xMax], 126, 174 )
        )
    
    print( "Total {n} image were created.".format( n=len( npy_label ) ) )
    
    return len( npy_label )


def pre_process( src_img ):
    print( "Image Pre-Processing ... ", flush=True, end="" )
    hsv = cv2.split( cv2.cvtColor( src_img, cv2.COLOR_BGR2HSV ) )
    
    # Hue, Saturation: Median( r = 5.0 )
    hsv[0] = cv2.medianBlur( hsv[0], 5 )
    hsv[1] = cv2.medianBlur( hsv[1], 5 )
    # Value: MeanShift( spacial=8.0 chroma=9.0 )
    hsv[2] = cv2.cvtColor( hsv[2], cv2.COLOR_GRAY2BGR )
    hsv[2] = cv2.pyrMeanShiftFiltering( hsv[2], 8, 18 )
    hsv[2] = cv2.cvtColor( hsv[2], cv2.COLOR_BGR2GRAY )
    
    print( "done! ", flush=True )
    
    return cv2.cvtColor( cv2.merge( hsv ), cv2.COLOR_HSV2BGR )


def get_roi( img, center_x, center_y, radius, copy=True ):
    if copy:
        return img.copy()[
            max( (center_y - radius), 0 ):min( (center_y + radius + 1), img.shape[0] ),
            max( (center_x - radius), 0 ):min( (center_x + radius + 1), img.shape[1] )
        ]
    else:
        return img[
           max( (center_y - radius), 0 ):min( (center_y + radius + 1), img.shape[0] ),
           max( (center_x - radius), 0 ):min( (center_x + radius + 1), img.shape[1] )
        ]


def get_answer( img_mask, img_answer ):
    assert img_mask.shape == img_answer.shape, "must be same shape!"
    
    White = np.array( [255, 255, 255], dtype=np.uint8 )
    Red = np.array( [0, 0, 255], dtype=np.uint8 )
    Black = np.array( [0, 0, 0], dtype=np.uint8 )
    
    dst = np.zeros( img_answer.shape, dtype=np.uint8 )
    
    dst[(img_mask == White).all(axis=2) & (img_answer == Red).all(axis=2)] = Red
    dst[(img_mask == White).all(axis=2) & (img_answer == Black).all(axis=2)] = White
    
    return dst
