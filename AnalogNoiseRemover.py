#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np


def median_1_X(img, wnd_size=9, pix_index='median'):
    """
    アナログ画像の白線ノイズ除去フィルタ

    - 以下の論文における「電線除去フィルタ」
      を参考にしている
        -- 王, 小野, et.al; 2009; 電子情報通信学会論文値 
          『時系列高さ画像の提案とそれを用いた
            車載カメラ画像と建物モデル間の対応付け』

    - 縦方向に長い 1 × 9 のメディアンフィルタにより白線ノイズ
      を除去する

    - パラメータ調整として、中央値ではなく何番目の値を採用するか
      を指定できるようにしている


    Parameters
    ----------
    img : numpy.ndarray
        除去対象の画像データ

    pix_index : [int, str]
        ソート済みのピクセル集合のうち
        採用する値のインデックス値
        中央値を採用する場合は pix_index='median' とする

    Returns
    -------
    filtered_img : numpy.ndarray
        フィルタ処理後の画像データ
    """

    assert type(img) == np.ndarray, \
        f"argument 'img' must be numpy.ndarray, not {type(img)}"
    assert type(wnd_size) == int, \
        f"argument 'int' must be numpy.ndarray, not {type(wnd_size)}"
    assert type(pix_index) in [int, str], \
        f"arguemnt 'pix_index' must be int or str('median'), not {type(pix_index)}"

    if type(pix_index) == str and pix_index not in ["median"]:
        print(f"Error! -- unknown parameters: pix_index='{pix_index}'")

    if type(pix_index) == int and pix_index >= wnd_size:
        print(f"Error! -- 'pix_index' must be smaller then 'wnd_size'")

    filtered_img = np.zeros(img.shape, dtype=img.dtype)

    print("Processing ... ")

    if img.ndim == 2:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                roi = img[:, j][i:i+wnd_size]
                if pix_index == 'median' or roi.shape[0] < wnd_size:
                    filtered_img[i, j] = np.median(roi).astype(img.dtype)
                else:
                    filtered_img[i, j] = np.sort(roi)[pix_index]

    elif img.ndim == 3:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(img.shape[2]):
                    roi = img[:, j, k][i:i+wnd_size]
                    if pix_index == 'median' or roi.shape[0] < wnd_size:
                        filtered_img[i, j, k] = np.median(
                            roi).astype(img.dtype)
                    else:
                        filtered_img[i, j, k] = np.sort(roi)[pix_index]

    print("done !")

    return filtered_img


def median_X_X(img, wnd_size=(3, 3), pix_index='median'):

    assert type(img) == np.ndarray, \
        f"argument 'img' must be numpy.ndarray, not {type(img)}"
    assert type(wnd_size) == tuple, \
        f"argument 'int' must be tuple, not {type(wnd_size)}"
    assert type(pix_index) in [int, str], \
        f"arguemnt 'pix_index' must be int or str('median'), not {type(pix_index)}"

    w, h = wnd_size

    if type(pix_index) == str and pix_index not in ["median"]:
        print(f"Error! -- unknown parameters: pix_index='{pix_index}'")

    if type(pix_index) == int and pix_index >= w * h:
        print(f"Error! -- 'pix_index' must be smaller then width * height ('wnd_size')")

    filtered_img = np.zeros(img.shape, dtype=img.dtype)

    print("Processing ... ")

    if img.ndim == 2:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                roi = img[i:i+h, j:j+w]
                if pix_index == 'median' or roi.size < w * h:
                    filtered_img[i, j] = np.median(
                        roi.flatten()).astype(img.dtype)
                else:
                    filtered_img[i, j] = np.sort(roi.flatten())[pix_index]

    elif img.ndim == 3:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(img.shape[2]):
                    roi = img[i:i+h, j:j+w, k]
                    if pix_index == 'median' or roi.size < w * h:
                        filtered_img[i, j, k] = np.median(
                            roi.flatten()).astype(img.dtype)
                    else:
                        filtered_img[i, j, k] = np.sort(
                            roi.flatten())[pix_index]

    print("done !")

    return filtered_img


if __name__ == '__main__':
    img_path = "./img/resource/aerial_roi1_raw.png"

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    roi_noise = img[515:530, :]
    # roi_denoise = median_X_X(roi_noise, (3, 3), pix_index=0)
    roi_denoise = median_1_X(roi_noise, wnd_size=3, pix_index=0)

    dst = img.copy()
    dst[515:530, :] = roi_denoise

    cv2.imshow("Result", dst)
    cv2.waitKey(0)

    cv2.imwrite("./img/resource/aerial_roi1_raw_denoised.png", dst)
