import numpy as np
import sys
import copy
import cv2
import SaveResultAsImage as sr
import CommonProcedures as cp
# for get random strings
import uuid
import random as rnd
from collections import deque

FLT_EPS = np.finfo( np.float32 ).eps

D_LOGGING = True
D_DEBUG_1 = False
D_DEBUG_2 = False
D_DEBUG_3 = False
# pass2() にてデバッグ用の結果を作成する場合、True
D_DEBUG_PASS2 = True
# calcEdgeFeatures() にてCannyエッジ処理を行う場合、True
# 通常はTrueにしておく
D_DEBUG_EXEC_CANNY = False


# FIXED (2017/09/13): 端点についての処理を追加
def calcEdgeFeatures( src_img ):
  # 入力画像のチェック
  assert (len( src_img.shape ) == 2), "\"src_img\" must be 1-ch grayscale edged image."
  
  if (D_DEBUG_EXEC_CANNY):
    edge_img = cv2.Canny( src_img, 126, 174 )
  else:
    edge_img = src_img
  
  features = {
    "total_length": 0, "length_average": 0, "angle_variance": 0, "endpoints": [], "branches": [],
    "passings"    : []
  }
  
  # 1: ENDPOINTS, 2: BRANCHES: 3: PASSINGS
  img_categorized = np.zeros( (src_img.shape[0], src_img.shape[1]), dtype=np.uint8)
  
  # 1st-Phase: 「端点」「分岐点」「通過点」の計算
  if D_LOGGING:
    print( "----- PASS 1 -----" )
  pass1( edge_img, features, img_categorized )
  
  # 2nd-Phase: 「エッジ長」の計算
  if D_LOGGING:
    print( "----- PASS 3 -----" )
  if len( features['passings'] ) != 0:
    pass3( features, img_categorized )
  
  # 3rd-Phase: エッジ分散値の計算
  # features['angle_variance'] = getEdgeAngleVariance( src_img )
  
  if (D_LOGGING):
    print( "total_length = %d, length_average = %lf, n_of_endpoints = %d, n_of_branches = %d, n_of_passings = %d" % (
      features['total_length'], features['length_average'], len( features['endpoints'] ), len( features['branches'] ),
      len( features['passings'] )),
           flush=True )
  
  if (D_DEBUG_1):
    print( features )
  
  return features


def isMatched( img_roi, array_profile ):
  bIsMatched = True
  
  assert (img_roi.shape[0] == array_profile.shape[0] and img_roi.shape[1] == array_profile.shape[1]), \
    "'img_roi'(%s) and 'array_profile'(%s) must be same size " % (str( img_roi.shape ), str( array_profile.shape ))
  
  img = (img_roi / 255).astype( np.uint8 )
  # for i in range( 0, array_profile.shape[0] ):
  #   for j in range( 0, array_profile.shape[1] ):
  # FIX(2018/02/13): Loop Optimize
  for (pp, rr) in zip(array_profile, img):
    for (p, r) in zip(pp, rr):
      if (p == -1):
        continue
      if (p != r):
        bIsMatched = False
        break
  
  return bIsMatched


# FIXED (2017/11/08): 再帰トラッキング→テンプレートマッチングに変更
# FIXED (2018/01/26): すべてテンプレートマッチングにより分類
def pass1( img, features, img_categorized ):
  assert (img.shape[0] == img_categorized.shape[0] and img.shape[1] == img_categorized.shape[1]), "must be same shape !"
  K_SIZE = 3
  # 端点を検出するためのテンプレート
  TEMP_ENDP = [np.array( [[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int8 ),
               np.array( [[0, 0, 0], [0, 1, 0], [-1, 1, -1]], dtype=np.int8 ),
               np.array( [[0, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.int8 ),
               ]
  # 分岐点を検出するためのテンプレート
  TEMP_BRNC = [np.array( [[-1, 1, -1], [1, 1, 1], [0, 0, 0]], dtype=np.int8 ),
               np.array( [[1, 0, 0], [0, 1, 0], [1, 0, 1]], dtype=np.int8 ),
               np.array( [[-1, 1, -1], [1, 1, 1], [-1, 1, -1]], dtype=np.int8 ),
               np.array( [[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.int8 )
               ]
  
  # FIX(2018/02/13): Init
  DICT_TEMP = { 'endpoints': [], 'branches': [] }
  for r in range(4):
    for x in TEMP_ENDP:
      DICT_TEMP['endpoints'].append(np.rot90(x, r))
    for x in TEMP_BRNC:
      DICT_TEMP['branches'].append(np.rot90(x, r))

  
  
  # 1Pass - Classifying 'endpoints' and 'passings'
  # FIX(2018/02/13): Loop Optimization

  for (i, xx) in enumerate( img ):
    for (j, x) in enumerate( xx ):
    
      if D_LOGGING:
        if (j == 0):
          print( f"PASS 1 - ({j}, {i})" )
    
      if (x != 255):
        continue
    
      img_roi = cp.getROI( img, j, i, 3 )
    
      features['total_length'] += 1
    
      isClassified = False
    
      # テンプレートによる分類：「端点」「分岐点」
      for (k, v) in DICT_TEMP.items():
        for edgeTemp in v:
          if isMatched( img_roi, edgeTemp ):
            features[k].append( [i, j] )
            # 1: ENDPOINTS, 2: BRANCHES: 3: PASSINGS
            if (k == 'endpoints'):
              img_categorized[i, j] = 1
            else:
              img_categorized[i, j] = 2
            isClassified = True
            if (D_DEBUG_1):
              print( "Classified! --> (%3d, %3d) at '%s'" % (j, i, k), flush=True )
    
      # マッチしなかったエッジ画素 →「通過点」
      if not isClassified:
        features['passings'].append( [i, j] )
        # 1: ENDPOINTS, 2: BRANCHES: 3: PASSINGS
        img_categorized[i, j] = 3
        if (D_DEBUG_1):
          print( "Classified! --> (%3d, %3d) at '%s'" % (j, i, 'passings'), flush=True )
  
  return


# エッジ線検出
# 'length_average' の計算
# FIXED(2018/02/13): cv2.connectedComponentsを使った形式に
def pass3( features, img_categorized ):
  
  # Img modify
  # 1: ENDPOINTS, 2: BRANCHES: 3: PASSINGS
  img = (img_categorized / 3).astype(np.uint8)*255
  
  # Labeling
  nLabels, labelImg = cv2.connectedComponents(img)
  
  features['length_average'] = np.count_nonzero(img) / nLabels
  
  if (D_DEBUG_PASS2):
    D_IMG = np.zeros( (img.shape[0], img.shape[1], 3), dtype=np.uint8 )

    COL = [ [0,0,0] ]
    COL += [ [rnd.randint( 64, 255 ), rnd.randint( 64, 255 ), rnd.randint( 64, 255 )] for _ in range(nLabels) ]
    
    for i in range(D_IMG.shape[0]):
      for j in range(D_IMG.shape[1]):
        D_IMG[i, j] = COL[labelImg[i, j]]
  
    cv2.imshow( "DEBUG_2", D_IMG )
    cv2.waitKey( 0 )
    cv2.imwrite( "img/DEBUG2_" + str( uuid.uuid4() ) + ".png", D_IMG )
  return


# img 全体の角度分散を計算
def getEdgeAngleVariance( img ):
  angle_hsv = getEdgeHSVArray( img )
  
  deg_arr = np.zeros( (360), dtype=np.float64 )
  total_votes = 0
  for i in range( 0, img.shape[0] ):
    for j in range( 0, img.shape[1] ):
      angle = angle_hsv[i, j, 0]
      strength = angle_hsv[i, j, 2]
      if (not (0 <= angle <= 360)):
        print( "data error! ( angle = %f )" % angle )
      if (not (strength < FLT_EPS)):
        # Voting
        deg_arr[int( angle ) % 360] += strength
        total_votes += strength
  
  retVal = np.sort( deg_arr )[::-1][0] / total_votes
  # print("Variance: ", retVal)
  return retVal


def getEdgeHSVArray( src_img ):
  assert (len( src_img.shape ) == 2), "\"src_img\" must be 1-ch grayscale image."
  dst = np.zeros( (src_img.shape[0], src_img.shape[1], 3), dtype=np.float32 )
  
  # print("Calculating Edge of whole image ... ", end="", flush=True)
  
  # print("sobel... ", end="", flush=True)
  dx = cv2.Sobel( src_img, cv2.CV_32F, 1, 0, ksize=3 )
  dy = cv2.Sobel( src_img, cv2.CV_32F, 0, 1, ksize=3 )
  
  # print("hsv calc... ", end="", flush=True)
  # 'Hue'        ::= Edge Angle ( normalized [0, 360] )
  dst[:, :, 0] = (np.arctan2( dy, dx ) + np.pi) * (180 / np.pi)
  # 'Satulation' ::= cannot use
  dst[:, :, 1] = 1.0
  # 'Value'      ::= Edge Strength = sqrt( dx^2 + dy^2 ) normalized [0, 1]
  s = np.sqrt( dx * dx + dy * dy )
  dst[:, :, 2] = s / s.max()
  
  # print("done!", flush=True)
  
  return dst


def extractFeature( npy_data, feat_name ):
  tmp = [0]
  for d in npy_data:
    tmp.append( d[1][feat_name] )
  
  return np.array( tmp )


def getEdgeFeatureAsImage( img_edge, npy_features ):
  # Col ::= [B, G, R]
  # dKeyCol = {'endpoints': [0, 255, 255], 'branches': [0, 0, 255], 'passings': [255, 128, 128]}
  dKeyCol = { 'passings': [255, 128, 128], 'branches': [0, 0, 255], 'endpoints': [0, 255, 255] }
  
  retImg = np.zeros( (img_edge.shape[0], img_edge.shape[1], 3), dtype=np.uint8 )
  
  for key in dKeyCol.keys():
    for p in npy_features[key]:
      retImg[p[0], p[1]] = dKeyCol[key]
  
  return retImg


########################
## 各種処理スクリプト ##
########################
def saveFeatureAsNPY( TARGET_NAME, N_OF_IMGS, SHOW_LOG=False ):
  assert (D_DEBUG_EXEC_CANNY == True), " this script needs D_DEBUG_EXEC_CANNY = True"
  
  result = []
  if (SHOW_LOG):
    print( "################################################", flush=True )
    print( "................................................", flush=True )
    print( "...########...########.....######.....########..", flush=True )
    print( "...##.........##.....##...##....##....##........", flush=True )
    print( "...##.........##.....##...##..........##........", flush=True )
    print( "...######.....##.....##...##...####...######....", flush=True )
    print( "...##.........##.....##...##....##....##........", flush=True )
    print( "...##.........##.....##...##....##....##........", flush=True )
    print( "...########...########.....######.....########..", flush=True )
    print( "................................................", flush=True )
    print( "################################################", flush=True )
    print( "## {0:42s} ##".format( "target: \"%s\"" % TARGET_NAME ), flush=True )
    print( "## {0:42s} ##".format( " total: %d" % N_OF_IMGS ), flush=True )
    print( "################################################", flush=True )
    print( "Processing ... ", end="", flush=True )
  
  for i in range( 1, N_OF_IMGS ):
    if (SHOW_LOG and (i % 5 == 0)):
      print( "%d ... " % i, end="", flush=True )
    
    file_path = "img/divided/" + TARGET_NAME + "/" + str.format( "%05d.png" % i )
    src_img = cv2.imread( file_path, cv2.IMREAD_GRAYSCALE )
    if (src_img is None):
      print( "file load error." )
    features = calcEdgeFeatures( src_img )
    features = {
      "length"        : features['length_average'],
      "endpoints"     : len( features['endpoints'] ) / features['total_length'],
      "branches"      : len( features['branches'] ) / features['total_length'],
      "passings"      : len( features['passings'] ) / features['total_length'],
      "angle_variance": features['angle_variance']
    }
    result.append( [i, features] )
  
  result = np.array( result )
  np.save( "data/" + TARGET_NAME + "_edge_feat.npy", result )
  print( "done! -- \"%s\"" % ("data/" + TARGET_NAME + "_edge_feat.npy") )
  
  return result


def saveFeatureAsCSV( TARGET_NAME, N_OF_IMGS, outToStdOut=False ):
  assert (D_DEBUG_EXEC_CANNY == True), " this script needs D_DEBUG_EXEC_CANNY = True"
  
  if not outToStdOut:
    result_path = 'data/' + TARGET_NAME
    f = open( result_path, 'w' )
  else:
    f = sys.stdout
  
  print( "TARGET_NAME: ", TARGET_NAME )
  print( "  N_OF_IMGS: ", N_OF_IMGS )
  
  f.write( "#,length_average,endpoints,branches,passings,angle_variance\n" )
  for i in range( 1, N_OF_IMGS ):
    file_path = "img/divided/" + TARGET_NAME + "/" + str.format( "%05d.png" % i )
    src_img = cv2.imread( file_path, cv2.IMREAD_GRAYSCALE )
    if (src_img is None):
      print( "file load error." )
    features = calcEdgeFeatures( src_img )
    features = {
      "length"        : features['length_average'],
      "endpoints"     : len( features['endpoints'] ) / features['total_length'],
      "branches"      : len( features['branches'] ) / features['total_length'],
      "passings"      : len( features['passings'] ) / features['total_length'],
      "angle_variance": features['angle_variance']
    }
    print( "%i,%3.6f,%3.6f,%3.6f,%3.6f,%3.6f" % (
      i,
      features['length'],
      features['endpoints'],
      features['branches'],
      features['passings'],
      features['angle_variance']) )
  
  if not outToStdOut:
    f.close()
    print( "saved result as \"%s\"" % result_path )


def proc3( src_img_path ):
  src_img = cv2.imread( src_img_path, cv2.IMREAD_GRAYSCALE )
  print( "start edge calc ... ", end="", flush=True )
  features = calcEdgeFeatures( src_img )
  print( "done !", flush=True )
  dst_img = getEdgeFeatureAsImage( src_img, features )
  # dst_img = cv2.resize(dst_img, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_NEAREST)
  # print("length: ", features['length_average'])
  
  cv2.imshow( "Test", dst_img )
  cv2.waitKey( 500 )
  cv2.imwrite( "img/DEBUG1_" + str( uuid.uuid4() ) + ".png", dst_img )
  return


def proc4( src_img_path, wnd_size, feature_name ):
  src_img = cv2.imread( src_img_path, cv2.IMREAD_COLOR )
  
  img_value = np.zeros( src_img.shape, dtype=np.uint8 )
  values = np.zeros( (0) )
  rects = []
  
  src_gray = cv2.cvtColor( src_img, cv2.COLOR_BGR2GRAY )
  
  for y in range( 0, src_img.shape[0], wnd_size ):
    for x in range( 0, src_img.shape[0], wnd_size ):
      if (0 <= y + wnd_size <= src_img.shape[0] and 0 <= x + wnd_size <= src_img.shape[1]):
        features = calcEdgeFeatures( src_gray[y:y + wnd_size, x:x + wnd_size] )
        values = np.append( values, features[feature_name] )
        rects.append( [x, y, x + wnd_size, y + wnd_size] )
  
  # values /= values.max()
  values = sr.percentileModification( values, 10 )
  for i in range( 0, len( values ) ):
    img_value = sr.drawRect( img_value, rects[i], values[i], rainbowColor=True )
  
  img = sr.createOverwrappedImage( src_img, img_value )
  cv2.imshow( "Test", img )
  cv2.waitKey( 0 )


def proc5( src_img_path ):
  src_img = cv2.imread( src_img_path, cv2.IMREAD_GRAYSCALE )
  
  hsv_array = getEdgeHSVArray( src_img )
  cv2.imshow( "Angle HSV", cv2.cvtColor( hsv_array, cv2.COLOR_HSV2BGR ) )
  cv2.waitKey( 0 )


def proc6( src_img_path, wnd_size, feature_name ):
  src_img = cv2.imread( src_img_path, cv2.IMREAD_COLOR )
  img_value = np.zeros( (src_img.shape[0], src_img.shape[1]), dtype=np.float64 )
  
  src_gray = cv2.cvtColor( src_img, cv2.COLOR_BGR2GRAY )
  
  for y in range( 0, src_img.shape[0] ):
    for x in range( 0, src_img.shape[1] ):
      rect = [(0 if (y - int( 0.5 * wnd_size ) < 0) else y - int( 0.5 * wnd_size )),
              (src_img.shape[0] if (y + int( 0.5 * wnd_size ) > src_img.shape[0]) else y + int( 0.5 * wnd_size )),
              (0 if (x - int( 0.5 * wnd_size ) < 0) else x - int( 0.5 * wnd_size )),
              (src_img.shape[1] if (x + int( 0.5 * wnd_size ) > src_img.shape[1]) else x + int( 0.5 * wnd_size ))]
      print( "rect: ", rect )
      features = calcEdgeFeatures( src_gray[rect[0]:rect[1], rect[2]:rect[3]] )
      img_value[y, x] = features[feature_name]
  
  for i in range( 0, img_value.shape[0] ):
    img_value[i] = sr.percentileModification( img_value[1], 10 ) * 255
  
  img_value = img_value.astype( np.uint8 )
  img_value = cv2.applyColorMap( img_value, cv2.COLORMAP_RAINBOW )
  print( "Done! " )
  cv2.imshow( "Test", img_value )
  cv2.waitKey( 0 )
  img = sr.createOverwrappedImage( src_img, img_value )
  cv2.imshow( "Test", img )
  cv2.waitKey( 0 )


if __name__ == '__main__':
  # proc3("img/test_set/00024.png")
  # proc3("img/test_set/00025.png")
  # proc3("/Users/popunbom/Google Drive/IDE_Projects/ImageResource/edge2.bmp")
  proc3("img/resource/edge2.bmp")
  # proc3( "img/resource/expr_1_aerial_customblur.png" )
  # proc3( "img/resource/expr_2_aerial_customblur.png" )
  # proc3("img/resource/expr_3_aerial.png")
  # proc3("img/divided/aerial_roi1_customeblur/canny_00026.png")
  # proc4("img/divided/new_roi1/00018.png", 7, 'angle_variance')
  # proc5("img/divided/new_roi1/00018.png")
  # proc6("img/divided/new_roi1/00018.png", 10, 'angle_variance')
  
  # TARGETS = ["new_roi", "new_blur_roi"]
  # N_OF_IMGS = [43, 27, 70]
  #
  # for i in [1, 2, 3]:
  #   for target in TARGETS:
  #     # saveFeatureAsCSV(target+str(i), N_OF_IMGS[i-1], outToStdOut=False)
  #     saveFeatureAsNPY(target+str(i), N_OF_IMGS[i-1], SHOW_LOG=True)
