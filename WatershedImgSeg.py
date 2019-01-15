#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# REF: https://qiita.com/ysdyt/items/5972c9520acf6a094d90

import os
from datetime import datetime
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

DIR_TMP_ROOT = 'img/tmp/watershed/' + datetime.now().strftime( "%Y%m%d_%H%M%S" )
D_SAVE_TMP_IMAGES = True


def equalizeHist( img, mode='hsv' ):
  """
  ヒストグラム平坦化を行う
  
  Parameters
  ----------
  img : numpy.ndarray
      入力画像データ。画像はグレースケールまたは RGB カラー画像である必要がある。
  mode : str
      動作切り替え用のパラメータ。デフォルト値は 'hsv'。
      mode='hsv' のとき、HSV色空間の V チャンネルに対してヒストグラム平坦化を行う
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
    f"""'mode' must be '{"' or '".join( ['hsv', 'rgb'] )}'"""
  if img.ndim == 2:
    return cv2.equalizeHist( img )
  else:
    if mode == 'hsv':
      img = cv2.cvtColor( img, cv2.COLOR_BGR2HSV )
      # V チャンネルのみを対象にする
      img[:, :, 2] = cv2.equalizeHist( img[:, :, 2] )
      img = cv2.cvtColor( img, cv2.COLOR_HSV2BGR )
    
    elif mode == 'rgb':
      for i in range( img.ndim ):
        img[:, :, i] = cv2.equalizeHist( img[:, :, i] )
    
    return img


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


def showImages( list_img, plt_title=None, list_title=None, list_cmap=None, tuple_shape=None ):
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
  assert list_cmap is None or len( [img for img in list_img if img.ndim == 2] ) == len( list_cmap ), \
    "Length of 'list_cmap' must be same as Num of gray-scale image in 'list_img'"
  assert list_title is None or len( list_img ) == len( list_title ), \
    "Length of 'list_title' must be same as Length of 'list_img'"
  assert tuple_shape is None or tuple_shape[0] * tuple_shape[1] >= len( list_img ), \
    " nrows * ncols of 'tuple_shape' must be equal or larger than Length of 'list_img'"
  
  plt.suptitle( plt_title )
  
  if tuple_shape is None:
    nrows = math.ceil( math.sqrt( len( list_img ) ) )
    ncols = math.ceil( len( list_img ) / nrows )
  else:
    ncols, nrows = tuple_shape
  
  for index, (img, title) in enumerate( zip( list_img, list_title ), start=1 ):
    axes = plt.subplot( ncols, nrows, index )
    axes.set_xticks( [] )
    axes.set_yticks( [] )
    axes.set_title( title )
    
    if img.ndim == 2:
      plt.imshow( img, cmap=(
        'gray' if list_cmap is None else list_cmap.pop( 0 )) )
    elif img.ndim == 3:
      plt.imshow( cv2.cvtColor( img, cv2.COLOR_BGR2RGB ) )
  
  plt.show()


def get_grayscaled_img( img ):
  """
  [Watershed] RGB カラー画像をグレースケール画像に変換する
  
  Parameters
  ----------
  img : numpy.ndarray
    RGB カラー画像
  Returns
  -------
  img_gs : numpy.ndarray
    グレースケール化された画像

  """
  img_gs = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
  # img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:,:,0]
  
  return img_gs

def get_binarized_img( img_gs ):
  """
  [Watershed] グレースケール画像を2値化画像に変換する
  Otsu 法による自動的な2値化を使用する
  
  Parameters
  ----------
  img_gs : numpy.ndarray
    グレースケール画像
    
  Returns
  -------
  thresh : int
    Otsu 法で求められた閾値
    
  img_bin : numpy.ndarray
    Otsu 法で求められた2値化画像

  """
  thresh, img_bin = cv2.threshold( img_gs, 0, 255, cv2.THRESH_OTSU )
  if np.sum( img_bin == 0 ) > np.sum( img_bin == 255 ):
    img_bin = cv2.bitwise_not( img_bin )
  
  return thresh, img_bin

def get_hole_filled_img( img_bin ):
  """
  [Watershed] 2値化画像の穴領域を削除する
  
  Parameters
  ----------
  img_bin : numpy.ndarray
    2値化画像

  Returns
  -------
  img_filled : numpy.ndarray
    処理によって穴領域が削除された2値化画像

  """
  kernel = np.array( [
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
  ], dtype=np.uint8 )
  
  img_filled = img_bin
  img_filled = cv2.morphologyEx( img_filled, cv2.MORPH_CLOSE, kernel, iterations=1 )
  img_filled = cv2.morphologyEx( img_filled, cv2.MORPH_OPEN, kernel, iterations=1 )
  
  return img_filled

def get_sure_bg( img_filled ):
  """
  [Watershed] 「確実な背景(sure background)」な画像を抽出する
  
  Parameters
  ----------
  img_filled : numpy.ndarray
    2値化画像
    前処理によって画像中の穴領域が削除されていることが前提

  Returns
  -------
  img_sure_bg : numpy.ndarray
    抽出された「確実な背景」な2値化画像

  """
  kernel = np.array( [
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
  ], dtype=np.uint8 )
  
  img_sure_bg = cv2.dilate( img_filled, kernel, iterations=1 )
  
  return img_sure_bg

def get_distance_img( img_filled ):
  """
  Parameters
  ----------
  img_filled : numpy.ndarray
    2値化画像
    前処理によって画像中の穴領域が削除されていることが前提
    
  Returns
  -------
  img_dist : numpy.ndarray
    距離画像(グレースケール)

  """
  
  img_dist = cv2.distanceTransform( img_filled, cv2.DIST_L2, 5 )
  
  return img_dist

def get_sure_fg( img_dist ):
  """
    [Watershed] 「確実な前景(sure foreground)」な画像を抽出する

    Parameters
    ----------
    img_dist : numpy.ndarray
      入力となる距離画像。グレースケール画像である必要がある。

    Returns
    -------
    img_sure_fg : numpy.ndarray
      抽出された「確実な前景」な2値化画像

    """
  thresh, img_sure_fg = cv2.threshold(
      img_dist, 0.5 * img_dist.max(), 255, 0 )
  # print(f"Threshold(Sure Foreground): {thresh}")
  
  img_sure_fg = img_sure_fg.astype( np.uint8 )
  
  return img_sure_fg

def get_img_unknown_area_img( img_sure_bg, img_sure_fg ):
  """
  
  Parameters
  ----------
  img_sure_bg : numpy.ndarray
    「確実な背景」を示す2値化画像
  img_sure_fg : numpy.ndarray
    「確実な前景」を示す2値化画像

  Returns
  -------
  img_unknown : numpy.ndarray
    前景でも背景でもない領域を示す2値化画像

  """
  img_unknown = cv2.subtract( img_sure_bg, img_sure_fg )
  
  return img_unknown


def watershed( img_path ):
  if D_SAVE_TMP_IMAGES:
    base_name = os.path.splitext( os.path.basename( img_path ) )[0]
  img = cv2.imread( img_path, cv2.IMREAD_COLOR )
  
  # 前処理: 平滑化
  img = cv2.medianBlur( img, 3 )
  
  # Step-1: グレースケール化
  img_gs = get_grayscaled_img( img )
  if D_SAVE_TMP_IMAGES:
    cv2.imwrite( os.path.join( DIR_TMP_ROOT, base_name + "_0_gs.png" ), img_gs )
  
  # Step-2: 2値化
  thresh, img_bin = get_binarized_img( img_gs )
  print( f"Threshold: {thresh}" )
  if D_SAVE_TMP_IMAGES:
    cv2.imwrite( os.path.join( DIR_TMP_ROOT, base_name + "_1_bin.png" ), img_bin )
  
  # Step-3: 画像の中の「穴」を埋める
  img_filled = get_hole_filled_img( img_bin )
  if D_SAVE_TMP_IMAGES:
    cv2.imwrite( os.path.join( DIR_TMP_ROOT, base_name + "_2_filled.png" ), img_filled )
  
  # Step-4: 確実な背景(sure background)の抽出
  img_sure_bg = get_sure_bg( img_filled )
  if D_SAVE_TMP_IMAGES:
    cv2.imwrite( os.path.join( DIR_TMP_ROOT, base_name + "_3_sure_bg.png" ), img_sure_bg )
  
  # Step-5: 距離画像の生成
  img_dist = get_distance_img(img_filled)
  if D_SAVE_TMP_IMAGES:
    cv2.imwrite( os.path.join( DIR_TMP_ROOT, base_name + "_4_distance.png" ),
                 (img_dist / img_dist.max() * 255.0).astype( np.uint8 ) )
  
  # Step-6: 確実な前景(sure foreground)の抽出
  img_sure_fg = get_sure_fg(img_dist)
  if D_SAVE_TMP_IMAGES:
    cv2.imwrite( os.path.join( DIR_TMP_ROOT, base_name + "_5_sure_fg.png" ), img_sure_fg )
   
  # Step-7: 前景でも背景でもない領域(unknown)の抽出
  img_unknown = get_img_unknown_area_img( img_sure_bg, img_sure_fg )
  if D_SAVE_TMP_IMAGES:
    cv2.imwrite( os.path.join( DIR_TMP_ROOT, base_name + "_6_img_unknown.png" ), img_unknown )
  
  # 各結果の確認
  showImages( [img_gs, img_bin, img_filled, img_dist, img_sure_fg, img_unknown],
              list_title=["Grayscale", "Binary", "Morph",
                          "Distance", "Foreground", "Unknown"],
              list_cmap=['gray', 'gray', 'gray', 'rainbow', 'gray', 'gray'],
              plt_title=os.path.basename( img_path ) )
  
  # Step-8: ラベリング
  ret, markers = cv2.connectedComponents( img_sure_fg )
  markers += 1
  markers[img_unknown == 255] = 0
  
  # Step-9: ラベリング結果に基づいて watershed を実施
  markers = cv2.watershed( img, markers )
  
  # 最終結果
  showImages( [img, img_bin, markers],
              list_title=["Input", "Binary", "Watershed"],
              list_cmap=['gray', 'rainbow'],
              tuple_shape=(1, 3),
              plt_title=os.path.basename( img_path )
              )
  
  return markers


if __name__ == '__main__':
  if D_SAVE_TMP_IMAGES:
    if not os.path.exists( DIR_TMP_ROOT ):
      os.mkdir( DIR_TMP_ROOT )
  
  for i in range( 100 ):
    watershed( f"img/divided/aerial_roi1/{i:05d}.png" )
  # watershed("/Users/popunbom/Google Drive/IDE_Projects/PyCharm/DmgAnalyzr/img/resource/aerial_blur_roi1.png")
