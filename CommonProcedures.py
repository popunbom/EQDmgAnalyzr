import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

import MaskLabeling as ml


def MAX( a, b ):
  return a if (a > b) else b


def MIN( a, b ):
  return a if (a < b) else b


def getRect( img_shape, points_of_label ):
  yMin = img_shape[0];
  xMin = img_shape[1]
  yMax = -1;
  xMax = -1
  
  for p in points_of_label:
    if (p == (-1, -1)):
      continue
    yMin = MIN( yMin, p[0] );
    yMax = MAX( yMax, p[0] )
    xMin = MIN( xMin, p[1] );
    xMax = MAX( xMax, p[1] )
  
  return [yMin, yMax, xMin, xMax]


def imageDivideByMask( src_img, npy_label, dir_name ):
  assert (src_img.shape[2] == 3), \
    " 'src_img' must be 3-ch RGB image."
  
  if (not os.path.exists( "img/divided/" + dir_name )):
    os.mkdir( "img/divided/" + dir_name )
  
  for i in range( len( npy_label ) ):
    P = getRect( src_img.shape, npy_label[i] )
    
    cv2.imwrite( "img/divided/" + dir_name + "/%05d.png" % i, src_img[P[0]:P[1], P[2]:P[3]] )
    cv2.imwrite( "img/divided/" + dir_name + "/canny_%05d.png" % i,
                 cv2.Canny( src_img[P[0]:P[1], P[2]:P[3]], 126, 174 ) )
  
  print( "Total %d image were created." % len( npy_label ) )
  
  return len( npy_label )


def imgPreProc( src_img ):
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


def getROI( img, center_x, center_y, kSize ):
  x = center_x;
  y = center_y
  assert (kSize % 2 != 0), "kSize is an odd number !"
  
  if len( img.shape ) == 2:
    ret = np.zeros( (kSize, kSize), dtype=img.dtype )
  else:
    ret = np.zeros( (kSize, kSize, img.shape[2]), dtype=img.dtype )
  
  for (m, i) in zip( range( 0, kSize ), range( -(kSize // 2), (kSize // 2) + 1 ) ):
    for (n, j) in zip( range( 0, kSize ), range( -(kSize // 2), (kSize // 2) + 1 ) ):
      if (0 <= y + i < img.shape[0] and 0 <= x + j < img.shape[1]):
        ret[m, n] = img[y + i, x + j]
  
  return ret


def getAnswer( img_mask, img_answer ):
  assert img_mask.shape == img_answer.shape, "must be same shape!"
  W = np.array( [255, 255, 255], dtype=np.uint8 )
  R = np.array( [0, 0, 255], dtype=np.uint8 )
  K = np.array( [0, 0, 0], dtype=np.uint8 )
  dst = np.zeros( img_answer.shape, dtype=np.uint8 )
  # for i in range(img_answer.shape[0]):
  #   for j in range(img_answer.shape[1]):
  #     a = img_answer[i][j]
  #     m = img_mask[i][j]
  #     if ((a == R).all() and (m == W).all()):
  #       dst[i][j] = R
  #     elif ((a == K).all() and (m == W).all()):
  #       dst[i][j] = W
  
  for i, (aa, mm) in enumerate( zip( img_answer, img_mask ) ):
    for j, (a, m) in enumerate( zip( aa, mm ) ):
      if i % 100 == 0 and j == 0: print( i, j )
      if (m == W).all():
        if (a == R).all():
          dst[i][j] = R
        elif (a == K).all():
          dst[i][j] = W
  
  return dst


def equalizeHistColored( img, mode='hsv' ):
  """
  ヒストグラム平坦化を行う(カラー画像対応)

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


def imshow( img, _cmap='gray', title="", show_axis=False ):
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
  if not show_axis:
    plt.tick_params( labelbottom=False, labelleft=False, labelright=False, labeltop=False,
                     bottom=False, left=False, right=False, top=False )
  if title:
    plt.suptitle(title)
    
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
  
  
  if tuple_shape is None:
    nrows = math.ceil( math.sqrt( len( list_img ) ) )
    ncols = math.ceil( len( list_img ) / nrows )
  else:
    ncols, nrows = tuple_shape

  # plt.figure( figsize=(ncols * 3, nrows * 1.7), dpi=150 )
  
  plt.suptitle( plt_title )
  
  for index, (img, title) in enumerate( zip( list_img, list_title ), start=1 ):
    axes = plt.subplot( ncols, nrows, index )
    axes.set_xticks( [] )
    axes.set_yticks( [] )
    axes.set_title( title )
    
    if img.ndim == 2:
      axes.imshow( img, cmap=(
        'gray' if list_cmap is None else list_cmap.pop( 0 )) )
    elif img.ndim == 3:
      axes.imshow( cv2.cvtColor( img, cv2.COLOR_BGR2RGB ) )
  
  plt.show()
  
  # Maximize Window
  mng = plt.get_current_fig_manager()
  # mng.window.state( 'zoomed' )  # TkAgg
  # mng.frame.Maximize( True )    # wxAgg
  # mng.window.showMaximized()    # Qt4Agg
  mng.resize(1920, 1080)        # macosx
  
  
  

if __name__ == '__main__':
    img = cv2.imread("./img/resource/aerial_roi1_raw.png", cv2.IMREAD_COLOR)
    imshow(img)
