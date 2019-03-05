#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# REF: https://qiita.com/ysdyt/items/5972c9520acf6a094d90

import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


def equalizeHist(img, mode='hsv'):
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
    assert img.ndim == 2 or img.ndim == 3, \
        "'img' must be Grayscale image or RGB Color image"
    assert mode in ['hsv', 'rgb'], \
        f"""'mode' must be '{"' or '".join(['hsv', 'rgb'])}'"""
    if img.ndim == 2:
        return cv2.equalizeHist(img)
    else:
        if mode == 'hsv':
            import pdb
            pdb.set_trace()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

        elif mode == 'rgb':
            for i in range(img.ndim):
                img[:, :, i] = cv2.equalizeHist(img[:, :, i])

        return img


def imshow(img, _cmap='gray'):
    """
    matplotlib を利用した画像表示をカンタンに行う関数
    
    Parameters
    ----------
    img : numpy.ndarray
        入力画像データ
        グレースケール画像を与えた場合、引数 '_cmap' を考慮する必要がある。
    _cmap : str
        グレースケール画像に対する疑似カラー処理のスタイルを指定する。
        デフォルト値は 'gray' (白黒のグレースケール表示)。
        指定するスタイル名の一覧は次を参照 : https://matplotlib.org/examples/color/colormaps_reference.html
        
    Returns
    -------

    """
    if img.ndim == 2:
        plt.imshow(img, cmap=_cmap)
    elif img.ndim == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


def showImages(list_img, plt_title=None, list_title=None, list_cmap=None, tuple_shape=None):
    """
    
    matplotlib の imshow を利用して、複数の画像をカンタンに表示するための関数
    
    Parameters
    ----------
    list_img : list[numpy.ndarray]
        入力画像データの配列(list)。
    plt_title : str
        画像全体のキャプション。デフォルト値は None。
    list_title : list[str]
        各画像に対するキャプション。デフォルト値は None。
        list_title で指定したキャプションは、list_img と同じインデックスの画像に対して
        キャプションがつく。
        list_title が None でない場合、list_title の配列長は list_img の配列長と同じ
        である必要がある。
    list_cmap : list[str]
        各グレースケール画像に対する疑似カラー処理のスタイル。デフォルト値は None。
        指定するスタイル名の一覧は次を参照 : https://matplotlib.org/examples/color/colormaps_reference.html
        list_cmap で指定したスタイルは、list_img からグレースケール画像のみを抽出した
        配列と同じインデックスの画像に対して適応される(詳細はソースコードを読むこと)。
        list_cmap が None でない場合、list_cmap の配列長は、list_img に含まれる
        グレースケール画像の数と同じでなければならない。
    tuple_shape : tuple[int, int]
        画像配置のタテとヨコの最大値を指定する。(タテ, ヨコ) の形で指定する。デフォルト値は None。
        デフォルト値 (None) の場合、表示する画像枚数から正方形に近くなるように自動的に計算される。
        tuple_shape が None でない場合、タテ × ヨコが画像枚数以上である必要がある。
        
        
    Returns
    -------

    """
    assert list_cmap is None or len([img for img in list_img if img.ndim == 2]) == len(list_cmap), \
        "Length of 'list_cmap' must be same as Num of gray-scale image in 'list_img'"
    assert list_title is None or len(list_img) == len(list_title), \
        "Length of 'list_title' must be same as Length of 'list_img'"
    assert tuple_shape is None or tuple_shape[0] * tuple_shape[1] >= len(list_img), \
        " nrows * ncols of 'tuple_shape' must be equal or larger than Length of 'list_img'"

    plt.suptitle(plt_title)

    if tuple_shape is None:
        nrows = math.ceil(math.sqrt(len(list_img)))
        ncols = math.ceil(len(list_img) / nrows)
    else:
        ncols, nrows = tuple_shape

    for index, (img, title) in enumerate(zip(list_img, list_title), start=1):
        axes = plt.subplot(ncols, nrows, index)
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_title(title)

        if img.ndim == 2:
            plt.imshow(img, cmap=(
                'gray' if list_cmap is None else list_cmap.pop(0)))
        elif img.ndim == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.show()


def watershed(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # img = equalizeHist(img)
    img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh, img_bin = cv2.threshold(img_gs, 0, 255, cv2.THRESH_OTSU)
    if np.sum(img_bin == 0) > np.sum(img_bin == 255):
        img_bin = cv2.bitwise_not(img_bin)

    # thresh, img_bin = 0, cv2.adaptiveThreshold(img_gs, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 1)

    # import pdb; pdb.set_trace()

    print(f"Threshold: {thresh}")

    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=np.uint8)

    img_morph = img_bin
    img_morph = cv2.morphologyEx(
        img_morph, cv2.MORPH_CLOSE, kernel, iterations=1)
    img_morph = cv2.morphologyEx(
        img_morph, cv2.MORPH_OPEN, kernel, iterations=1)

    sure_bg = img_morph

    dist_transform = cv2.distanceTransform(img_morph, cv2.DIST_L2, 5)

    thresh, sure_fg = cv2.threshold(
        dist_transform, 0.5 * dist_transform.max(), 255, 0)
    # print(f"Threshold(Sure Foreground): {thresh}")

    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)

    showImages([img_gs, img_bin, img_morph, dist_transform, sure_fg, unknown],
               list_title=["Grayscale", "Binary", "Morph",
                           "Distance", "Foreground", "Unknown"],
               plt_title=os.path.basename(img_path))
    ret, markers = cv2.connectedComponents(sure_fg)
    markers += 1

    markers[unknown == 255] = 0

    # imshow(markers, _cmap='terrain')

    markers = cv2.watershed(img, markers)
    # imshow(markers, _cmap='rainbow')
    showImages([img, img_bin, markers],
               list_title=["Input", "Binary", "Watershed"],
               list_cmap=['gray', 'rainbow'],
               tuple_shape=(1, 3),
               plt_title=os.path.basename(img_path)
               )


if __name__ == '__main__':
    for i in range(100):
        watershed(f"img/divided/aerial_roi1/{i:05d}.png")
    # watershed("/Users/popunbom/Google Drive/IDE_Projects/PyCharm/DmgAnalyzr/img/resource/aerial_blur_roi1.png")
