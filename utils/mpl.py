#/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: popunbom <fantom0779@gmail.com>
# Created At: 2019/08/29
import math

import cv2
import numpy as np
import matplotlib.pyplot as plt


def imshow( img, _cmap='gray' ):
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
        plt.imshow( img, cmap=_cmap )
    elif img.ndim == 3:
        plt.imshow( cv2.cvtColor( img, cv2.COLOR_BGR2RGB ) )
    plt.show()


def show_images( list_img, plt_title=None, list_title=None, list_cmap=None, tuple_shape=None, fig_size=(6, 6), logger=None ):
    """

    matplotlib の imshow を利用して、複数の画像をカンタンに表示するための関数

    Parameters
    ----------
    list_img : list[numpy.ndarray]
        入力画像データの配列(list)。
    fig_size : tuple of int
        画像1枚あたりの大きさ (幅[inch], 高さ[inch])
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
        画像配置の行数と列数の最大値を指定する。(行数, 列数) の形で指定する。デフォルト値は None。
        デフォルト値 (None) の場合、表示する画像枚数から正方形に近くなるように自動的に計算される。
        tuple_shape が None でない場合、行数 × 列数が画像枚数以上である必要がある。
    logger : ImageLogger
        ImageLogger インスタンス
        指定された場合、画像を個別にファイル出力する

    Returns
    -------

    """
    assert list_cmap is None or len( [img for img in list_img if img.ndim == 2] ) == len( list_cmap ), \
        "Length of 'list_cmap' must be same as Num of gray-scale image in 'list_img'"
    assert list_title is None or len( list_img ) == len( list_title ), \
        "Length of 'list_title' must be same as Length of 'list_img'"
    assert tuple_shape is None or tuple_shape[0] * tuple_shape[1] >= len( list_img ), \
        " nrows * ncols of 'tuple_shape' must be equal or larger than Length of 'list_img'"
    
    
    # plt.suptitle( plt_title )
    
    if tuple_shape is None:
        nrows = math.ceil( math.sqrt( len( list_img ) ) )
        ncols = math.ceil( len( list_img ) / nrows )
    else:
        nrows, ncols = tuple_shape

    w, h = fig_size
    fig_size = (ncols * w, nrows * h)
    fig, _ax = plt.subplots( nrows, ncols, figsize=fig_size, squeeze=False )
    fig.suptitle( plt_title )

    
    ax = _ax.flatten()
    for i, (img, title) in enumerate( zip( list_img, list_title ) ):
        ax[i].set_xticks( [] )
        ax[i].set_yticks( [] )
        ax[i].set_title( title )
        
        if img.ndim == 2:
            cmap = 'gray' if list_cmap is None else list_cmap.pop( 0 )
            ax[i].imshow(
                img,
                cmap=cmap
            )
            
        elif img.ndim == 3:
            ax[i].imshow( cv2.cvtColor( img, cv2.COLOR_BGR2RGB ) )
            
        
        if logger is not None:
            logger.logging_img(
                img,
                file_name=f"{plt_title} - {title}",
                cmap=cmap
            )

    plt.show()
