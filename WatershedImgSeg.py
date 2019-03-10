#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# REF: https://qiita.com/ysdyt/items/5972c9520acf6a094d90

import os

from datetime import datetime
import numpy as np
import cv2


from CommonProcedures import equalizeHist, imshow, showImages


DIR_TMP_ROOT = 'img/tmp/watershed/' + datetime.now().strftime( "%Y%m%d_%H%M%S" )
D_SAVE_TMP_IMAGES = True


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
  
  
  - Otsu 法による自動的な2値化を使用する
  
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
  [Watershed] 距離画像を作成する
  
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
  [Watershed] 前景でも背景でもない「不明な領域(unknown)」を抽出する
  
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
  """
  [Watershed] メインルーチン
  
  Parameters
  ----------
  img_path : str
    画像へのパス
  
  Returns
  -------
  markers : numpy.ndarray
    Watershed により領域分割された結果画像 (ラベリング画像)
  """
  
  
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
