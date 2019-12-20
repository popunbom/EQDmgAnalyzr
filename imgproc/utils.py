# /usr/bin/env python3
# -*- coding: utf-8 -*-

# imgproc/utils.py: 各モジュールで共通する処理
# Author: popunbom <fantom0779@gmail.com>
# Created At: 2019/06/21
import functools
import os
import re
from multiprocessing import current_process
from multiprocessing.pool import Pool

import cv2
import numpy as np

from math import ceil

from skimage.morphology import disk
from tqdm import tqdm, trange

from utils.assertion import TYPE_ASSERT, SAME_SHAPE_ASSERT, NDIM_ASSERT, NDARRAY_ASSERT
from utils.common import n_args, eprint

import scipy.ndimage as ndi

"""
imgproc/utils.py : 汎用的な画像処理
"""


def compute_by_window(imgs, func, window_size=16, step=2, dst_dtype=np.float32, n_worker=12):
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
    dst_dtype : type, default numpy.float32
        返却値のデータ型
    n_worker : int, default 4
        並列するプロセス数

    Returns
    -------
    numpy.ndarray
        各切り取り画像に対する処理結果の行列
    """
    
    # TYPE ASSERTION
    TYPE_ASSERT(imgs, [np.ndarray, tuple])
    TYPE_ASSERT(window_size, [int, tuple])
    TYPE_ASSERT(step, [int, tuple])
    TYPE_ASSERT(dst_dtype, type)
    TYPE_ASSERT(n_worker, int)
    
    if isinstance(imgs, np.ndarray):
        imgs = tuple([imgs])
    
    for img in imgs:
        TYPE_ASSERT(img, np.ndarray)
    for i in range(len(imgs) - 1):
        SAME_SHAPE_ASSERT(imgs[i], imgs[i + 1])
    
    n_imgs = len(imgs)
    height, width = imgs[0].shape[:2]

    assert callable(func) and n_args(func) >= n_imgs, \
        "argument 'func' must be callable object which has {0} argument at least. \n".format(n_imgs) + \
        "  ( num of argumets of 'func' depends on argument 'imgs')"
    
    if isinstance(step, int):
        step = tuple([step] * 2)
    if isinstance(window_size, int):
        window_size = tuple([window_size] * 2)
    
    s_i, s_j = step
    w_w, w_h = window_size
    results_shape = ceil(height / s_i), ceil(width / s_j)
    
    # Add padding to input images
    eprint("Add padding ... ")
    imgs = [
        np.pad(
            img,
            pad_width=[
                tuple([w_w // 2]),
                tuple([w_h // 2]),
            ] + [] if img.ndim == 2 else [tuple([0])],
            mode="constant",
            constant_values=0
        )
        for img in imgs
    ]

    
    if n_worker == 1:
        results = np.ndarray(results_shape, dtype=dst_dtype)
        
        for ii, i in tqdm(enumerate(range(w_h // 2, height + w_h // 2, s_i)), total=results_shape[0]):
            
            for jj, j in tqdm(enumerate(range(w_w // 2, width + w_w // 2, s_j)), total=results_shape[1], leave=False):
                
                rois = [
                    img[
                        get_window_rect(
                            img.shape,
                            center=(j, i),
                            wnd_size=(w_w, w_h),
                            ret_type="slice"
                        )
                    ]
                    for img in imgs
                ]
                
                results[ii][jj] = func(*rois)
    
    else:
        global _func
        global _callee
        
        _func = func
        
        def _callee(_imgs, _func, _width, _s_j, _w_w, _n_loop):
            
            _worker_id = int(re.match(r"(.*)-([0-9]+)$", current_process().name).group(2))
            _desc = f"Worker #{_worker_id:3d}"
            
            _results = list()
            
            for jj, j in tqdm(enumerate(range(_w_w // 2, _width + _w_w // 2, _s_j)), total=_n_loop, desc=_desc,
                              position=_worker_id,
                              leave=False):
                _rois = [
                    # _roi[:, j:j + _w_w]
                    _roi[
                        get_window_rect(
                            _roi.shape,
                            center=(j, -1),
                            wnd_size=(_w_w, -1),
                            ret_type="slice"
                        )
                    ]
                    for _roi in _imgs
                ]
                _results.append(_func(*_rois))
            
            return _results
        
        
        progress_bar = tqdm(total=results_shape[0])
        
        def _update_progressbar(arg):
            progress_bar.update()
        
        pool = Pool(processes=n_worker, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
        
        results = list()
        for ii, i in enumerate(range(w_h // 2, height + w_h // 2, s_i)):
            
            rois = [
                img[
                    get_window_rect(
                        img.shape,
                        center=(-1, i),
                        wnd_size=(-1, w_h),
                        ret_type="slice"
                    )
                ]
                for img in imgs
            ]
            
            results.append(
                pool.apply_async(
                    _callee,
                    args=(rois, func, width, s_j, w_h, results_shape[1]),
                    callback=_update_progressbar
                )
            )
        pool.close()
        pool.join()
        
        results = np.array(
            [result.get() for result in results],
            dtype=dst_dtype
        )
    
    return results


def get_rect(img_shape, points):
    """
    領域を囲む矩形の座標を計算する
    
    Parameters
    ----------
    img_shape : tuple of int
        画像サイズ (高さ, 幅)
    points : list of points or array of points
        領域座標のリスト
    
    Returns
    -------
    list of tuple of int
        領域を囲む矩形の左上, 右下の座標
        [(yMin, xMin), (yMax, xMax)]
    """
    assert isinstance(img_shape, tuple), "'img_shape' must be tuple"
    
    yMax, xMax = 0, 0
    yMin, xMin = img_shape
    
    points = np.array(points, dtype=np.int32)
    
    yMin = min(np.min(points[:, 0]), yMin)
    xMin = min(np.min(points[:, 1]), xMin)
    yMax = max(np.max(points[:, 0]), yMax)
    xMax = max(np.max(points[:, 1]), xMax)
    
    if yMax <= yMin or xMax <= xMin:
        raise ValueError("Value Range Error: yMax <= yMin || xMax <= xMin")
    
    # return [yMin, yMax, xMin, xMax]
    return [(yMin, xMin), (yMax, xMax)]


def get_window_rect(img_shape, center, wnd_size, ret_type="tuple"):
    """
    中心座標から一定の大きさの矩形を切り出す
    
    Parameters
    ----------
    img_shape : tuple of ints
        切り出す画像の形状
        - ndarray.shape である必要がある
    center : tuple of ints
        中心座標: (x, y)
    wnd_size : int or tuple of ints
        切り出す矩形の大きさ
        負値の場合、全体を切り出す
        tuple の場合は (幅, 高さ)
    ret_type : string
        返却値の種類を指定
        - `tuple` と `slice` が使用可能

    Returns
    -------
    tuple of ints or tuple of slice
    - ret_type が `tuple` の場合、
      (左上のx座標、左上のy座標, 右下のx座標、右下のy座標)
      のタプル
    - ret_type が `slice` の場合、ndarray のスライス
    """
    
    TYPE_ASSERT(img_shape, tuple)
    TYPE_ASSERT(center, tuple)
    TYPE_ASSERT(wnd_size, [int, tuple])
    TYPE_ASSERT(ret_type, str)
    
    assert ret_type in ("tuple", "slice"), \
        "`ret_type` must be 'tuple' or 'slice'"
    
    if isinstance(wnd_size, int):
        wnd_size = tuple([wnd_size] * 2)
    
    height, width = img_shape[:2]
    c_x, c_y = center
    w_w, w_h = wnd_size
    
    if ret_type == "tuple":
        tl_y = 0 if w_h < 0 else max(0, int(c_y - w_h // 2))
        tl_x = 0 if w_w < 0 else max(0, int(c_x - w_w // 2))
        br_y = img_shape[0] if w_h < 0 else min(height, int(c_y + w_h // 2))
        br_x = img_shape[1] if w_w < 0 else min(width, int(c_x + w_w // 2))
        
        return tuple([tl_x, tl_y, br_x, br_y])
    
    elif ret_type == "slice":
        return np.s_[
            slice(None) if w_h < 0 else slice(c_y - (w_h // 2), c_y + (w_h // 2 + 1)),
            slice(None) if w_w < 0 else slice(c_x - (w_w // 2), c_x + (w_w // 2 + 1))
        ]
        # if w_h < 0 and w_w < 0:
        #     return np.s_[:, :]
        # elif w_h < 0:
        #     return np.s_[
        #         :,
        #         c_x - (w_w // 2):c_x + (w_w // 2 + 1)
        #     ]
        # elif w_w < 0:
        #     return np.s_[
        #         c_y - (w_h // 2):c_y + (w_h // 2 + 1),
        #         :
        #     ]
        # else:
        #     return np.s_[
        #         c_y - (w_h // 2):c_y + (w_h // 2 + 1),
        #         c_x - (w_w // 2):c_x + (w_w // 2 + 1)
        #     ]
    else:
        return None


def disk_mask(r, h, w):
    """
    円盤状のマスク画像を生成する

    Parameters
    ----------
    r : float
        半径
    h, w : int
        生成される画像の大きさ

    Returns
    -------
    mask : numpy.ndarray
        円盤状マスク

    Notes
    -----
    `mask`
        - 1-Bit (bool 型) 2値画像
        - 中心から半径 r の部分が黒(0)、それ以外が白(1)
    """
    
    TYPE_ASSERT(r, [int, float])
    TYPE_ASSERT(h, int)
    TYPE_ASSERT(w, int)
    
    mask = disk(r)
    p_h, p_w = (h - mask.shape[0], w - mask.shape[1])
    
    mask = np.pad(
        mask,
        [(
            (p_h) // 2,
            (p_h) // 2 + (p_h % 2)
        ), (
            (p_w) // 2,
            (p_w) // 2 + (p_w % 2)
        )],
        'constant'
    ).astype(bool)

    return mask

def divide_by_mask(img, npy_label, dir_name):
    """
    マスク画像をもとに画像分割を行う
    
    Parameters
    ----------
    img : numpy.ndarray
        入力画像
    npy_label
        MaskLabeling.getMaskLabel によるマスク処理結果
    dir_name
        分割画像のフォルダ名
    Returns
    -------
    int
        生成された画像枚数
    """
    TYPE_ASSERT(img, np.ndarray)
    TYPE_ASSERT(npy_label, np.ndarray)
    TYPE_ASSERT(dir_name, str)
    
    if (not os.path.exists("img/divided/" + dir_name)):
        os.mkdir("img/divided/" + dir_name)
    
    if img.ndim != 3:
        img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gs = img
    
    for i in range(len(npy_label)):
        (yMin, xMin), (yMax, xMax) = get_rect(img.shape, npy_label[i])
        
        cv2.imwrite(
            "img/divided/{dir_name}/{file_name:05d}.png".format(
                dir_name=dir_name, file_name=i), img[yMin:yMax, xMin:xMax])
        cv2.imwrite(
            "img/divided/{dir_name}/canny_{file_name:05d}.png".format(
                dir_name=dir_name, file_name=i),
            cv2.Canny(img_gs[yMin:yMax, xMin:xMax], 126, 174))
    
    print("Total {n} image were created.".format(n=len(npy_label)))
    
    return len(npy_label)


def check_if_binary_image(img):
    """
    画像が2値画像かどうか判定する
    
    - numpy.unique によって、行列内の要素値
      を重複なしに取得できることを利用する
    
    Parameters
    ----------
    img : numpy.ndarray
        入力画像 (グレースケール画像)

    Returns
    -------
    bool
        2値画像かどうか
    """
    
    TYPE_ASSERT(img, np.ndarray)
    
    return img.ndim == 2 and np.unique(img).size == 2


def get_roi(img, center_x, center_y, radius, copy=True):
    """
    中心座標, 半径をもとに ROI を取得する
    
    Parameters
    ----------
    img : numpy.ndarray
        入力画像
    center_x, center_y : int
        中心座標
    radius : int
        半径
    copy : bool
        画像データをコピーするかどうか

    Returns
    -------
    numpy.ndarray
        ROI 画像

    """
    TYPE_ASSERT(img, np.ndarray)
    TYPE_ASSERT(center_x, int)
    TYPE_ASSERT(center_y, int)
    TYPE_ASSERT(radius, int)
    TYPE_ASSERT(copy, bool)
    
    if copy:
        return img.copy()[max((center_y - radius), 0):min((center_y + radius +
                                                           1), img.shape[0]),
               max((center_x - radius), 0):min((center_x + radius +
                                                1), img.shape[1])]
    else:
        return img[max((center_y - radius), 0):min((center_y + radius +
                                                    1), img.shape[0]),
               max((center_x - radius), 0):min((center_x + radius +
                                                1), img.shape[1])]


def gen_overlay_by_gt(img_mask, img_gt):
    """
    正解データと建物マスク画像のオーバーレイ画像の生成
    
    Parameters
    ----------
    img_mask : numpy.ndarray
        マスク画像
    img_gt : numpy.ndarray
        正解データ

    Returns
    -------
    numpy.ndarray
        オーバーレイ画像
    """
    TYPE_ASSERT(img_mask, np.ndarray)
    TYPE_ASSERT(img_gt, np.ndarray)
    SAME_SHAPE_ASSERT(img_mask, img_gt)
    
    White = np.array([255, 255, 255], dtype=np.uint8)
    Red = np.array([0, 0, 255], dtype=np.uint8)
    Black = np.array([0, 0, 0], dtype=np.uint8)
    
    dst = np.zeros(img_gt.shape, dtype=np.uint8)
    
    dst[(img_mask == White).all(axis=2) & (img_gt == Red).all(axis=2)] = Red
    dst[(img_mask == White).all(axis=2)
        & (img_gt == Black).all(axis=2)] = White
    
    return dst


def zoom_to_img_size(img, shape):
    """
    画像と同じ大きさになるように拡大する
    
    - `img` を `shape` で指定したサイズに拡大する
    
    Parameters
    ----------
    img : numpy.ndarray
        入力画像
    shape : tuple of int
        画像サイズ

    Returns
    -------
    numpy.ndarray
        拡大された画像
    """
    
    NDARRAY_ASSERT(img)
    TYPE_ASSERT(shape, tuple)
    
    if img.shape[:2] == shape:
        return img
    
    return ndi.zoom(
        img,
        (shape[0] / img.shape[0], shape[1] / img.shape[1]),
        order=0,
        mode='nearest'
    )


def apply_road_mask(img, mask, bg_value=0):
    NDARRAY_ASSERT(img)
    NDARRAY_ASSERT(mask, ndim=2)
    SAME_SHAPE_ASSERT(img, mask, ignore_ndim=True)

    assert check_if_binary_image(mask), \
        "'mask' must be binary image"

    masked = img.copy()
    if img.ndim == 2:
        masked[mask == mask.max()] = bg_value.astype(masked.dtype)
    else:
        masked[mask == mask.max(), :] = np.full(
            masked.shape[2],
            fill_value=bg_value,
            dtype=masked.dtype
        )

    return masked
