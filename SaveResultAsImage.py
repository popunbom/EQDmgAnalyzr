import numpy as np
import cv2
import EdgeProfiler as ep
from imgproc.labeling import labeling_from_mask


def getMagnitudeColor( strength ):
  assert (0.0 <= strength and strength <= 1.0), "arguments must be [0,1]"
  
  i = strength * 360
  
  if (0 <= i and i < 60):
    col = np.array( [i / 60, 0, 0] )
  elif (60 <= i and i < 120):
    col = np.array( [1.0, (i - 60) / 60, 0] )
  elif (120 <= i and i < 180):
    col = np.array( [1.0 - (i - 120) / 60, 1.0, 0] )
  elif (180 <= i and i < 240):
    col = np.array( [0, 1.0, (i - 180) / 60] )
  elif (240 <= i and i < 300):
    col = np.array( [0, 1.0 - (i - 240) / 60, 1.0] )
  elif (300 <= i and i <= 360):
    col = np.array( [(i - 300) / 60, (i - 300) / 60, 1.0] )
  else:
    print( "range error! ", flush=True )
    col = np.array( [0, 0, 0] )
  
  return np.array( col * 255, dtype=np.uint8 )


def createOverwrappedImage( img_src, data_img, doGSConv=True ):
  assert (img_src.shape[2] == 3), "img_src must be 3-ch color image."
  alpha = 0.70
  # dst_img = cv2.add(img_src, data_img)
  if (doGSConv):
    img_src = cv2.cvtColor( img_src, cv2.COLOR_BGR2GRAY )
    img_src = cv2.cvtColor( img_src, cv2.COLOR_GRAY2BGR )
  
  dst_img = cv2.addWeighted( img_src, alpha, data_img, (1 - alpha), 0 )
  return dst_img


def percentileModification( data, p ):
  ret = np.zeros( data.shape, dtype=np.float32 )
  
  print( "Do Percentile Modification ... ", end="", flush=True )
  
  P_MAX = int( 100 / p );
  P = []
  for i in range( P_MAX ):
    P.append( np.percentile( data, (i + 1) * p ) )
  
  for i in range( data.shape[0] ):
    j = 0
    while ((j < len( P )) and (P[j] < data[i])):
      j += 1
    ret[i] = j / P_MAX
  
  print( "done!", flush=True )
  return ret


def drawRect( img, rect, value, rainbowColor=False ):
  assert (img.shape[2] == 3), "\"img\" must be 3ch BGR color image."
  assert (0 <= value <= 1), "\"value\" must be in range of [0, 1]"
  assert (len( rect ) == 4), "\"rect\" has 4 parameter: [X, Y, X+W, Y+H]"
  
  if (rainbowColor):
    col = getMagnitudeColor( value )
  else:
    col = [int( 255 * value ), 0, int( 255 * value )]
  
  for y in range( rect[1], rect[3] ):
    for x in range( rect[0], rect[2] ):
      img[y, x] = col
  
  return img


def createResultImg( img_src, npy_data, npy_label, raibowColor=False ):
  assert (npy_label.shape == npy_data.shape), "Label array and Feature data-array must be same size"
  assert (0 <= npy_data.min() and npy_data.max() <= 1), "data array must be in range of [0, 1]"
  
  print( "Making output image ... ", end="", flush=True )
  label_img = np.ones( (img_src.shape[0], img_src.shape[1], 3), np.uint8 ) * 255
  
  for i in range( 1, len( npy_data ) ):
    for p in npy_label[i]:
      if (raibowColor):
        label_img[p[0], p[1]] = getMagnitudeColor( npy_data[i] )
      else:
        label_img[p[0], p[1]] = [int( 255 * npy_data[i] ), 0, int( 255 * npy_data[i] )]
  
  print( "done!", flush=True )
  return label_img


def extractInRange( npy_data, q_min, q_max ):
  min = np.percentile( npy_data, q_min )
  max = np.percentile( npy_data, q_max )
  
  for i in range( npy_data.shape[0] ):
    npy_data[i] = 1.0 if (min <= npy_data[i] <= max) else 0.0
  
  return npy_data


def saveEdgeFeatureAsImage( src_img, mask_img, npy_features, feat_name, SHOW_LOG=False, SHOW_IMG=False,
                            doDataMod=True ):
  assert (src_img.shape[0] == mask_img.shape[0] and src_img.shape[1] == mask_img.shape[1]), \
    " 'src_img' and 'mask_img' must be the same size."
  assert (src_img.shape[2] == 3), \
    " 'src_img' must be 3-ch RGB image."
  assert (len( mask_img.shape ) == 2), \
    " 'src_img' must be 1-ch Grayscale image"
  
  # NPY Data Load
  npy_label = labeling_from_mask( mask_img )

  if SHOW_LOG:
    print( "  feat: %s" % feat_name, flush=True )
  
  # Extract specifical feature
  npy_data = ep.extractFeature( npy_features, feat_name )
  
  # Data Modification
  if doDataMod:
    npy_data = percentileModification( npy_data, 1 )
  else:
    npy_data = npy_data / npy_data.max()
  
  # for debug
  for i in range( 1, len( npy_data ) ):
    print( npy_data[i] )
  
  # Create Image
  dst_img = createResultImg( src_img, npy_data, npy_label, raibowColor=True )
  dst_img = createOverwrappedImage( src_img, dst_img, doGSConv=True )
  
  if SHOW_IMG:
    cv2.imshow( "Test", dst_img )
    cv2.waitKey( 0 )
  return dst_img


def clasify( src_img, mask_img, npy_features, npy_thresholds, SHOW_LOG, SHOW_IMG ):
  assert (src_img.shape[0] == mask_img.shape[0] and src_img.shape[1] == mask_img.shape[1]), \
    " 'src_img' and 'mask_img' must be the same size."
  assert (src_img.shape[2] == 3), \
    " 'src_img' must be 3-ch RGB image."
  assert (len( mask_img.shape ) == 2), \
    " 'src_img' must be 1-ch Grayscale image"
  
  features = ['length', 'endpoints', 'branches', 'passings']
  
  # NPY Data Load
  npy_label = labeling_from_mask( mask_img )
  
  roi_category = { }
  
  for feature in features:
    # Extract specify feature
    npy_data = ep.extractFeature( npy_features, feature )
    npy_data = percentileModification( npy_data, 1 )
    
    # Extract specify thresholds
    thresh = npy_thresholds[feature]['th']
    
    # Add list to dict
    roi_category.update( { feature: np.ndarray( (len( npy_data )), dtype=np.uint8 ) } )
    
    # Clasify by thresholds
    # 0 - Collapsed completely
    # 1 - Collapsed roughly
    # 2 - Not Collapsed
    ### Thresholds
    #  npy_thresholds = { 'feat_name', {'type: 0 or 1, th:[]} }:
    # if type == 0
    ##     0.0 < p < th[0]: Not Collapsed (2)
    ##   th[1] < p < th[1]: Collapsed roughly (1)
    ##   th[1] < p <   1.0: Collapsed completely (0)
    # elif type == 1
    ##     0.0 < p < th[0]: Collapsed completely (0)
    ##   th[1] < p < th[1]: Collapsed roughly (1)
    ##   th[1] < p <   1.0: Not Collapsed (2)
    
    for i in range( 1, len( npy_data ) ):
      # 0 - Collapsed completely
      # 1 - Collapsed roughly
      # 2 - Not Collapsed
      if npy_thresholds[feature]['type'] == 0:
        if (0.0 <= npy_data[i] <= thresh[0]):
          roi_category[feature][i] = 2
        elif (thresh[0] < npy_data[i] < thresh[1]):
          roi_category[feature][i] = 1
        elif (thresh[1] <= npy_data[i] <= 1.0):
          roi_category[feature][i] = 0
      # 0 - Collapsed completely
      # 1 - Collapsed roughly
      # 2 - Not Collapsed
      elif npy_thresholds[feature]['type'] == 1:
        if (0.0 <= npy_data[i] <= thresh[0]):
          roi_category[feature][i] = 0
        elif (thresh[0] < npy_data[i] < thresh[1]):
          roi_category[feature][i] = 1
        elif (thresh[1] <= npy_data[i] <= 1.0):
          roi_category[feature][i] = 2
  
  npy_result = np.ndarray( (len( npy_label )), dtype=np.uint8 )
  for i in range( 1, len( npy_data ) ):
    vote = np.zeros( (3), dtype=np.uint8 )
    for feature in features:
      vote[roi_category[feature][i]] += 1
    npy_result[i] = vote.argmax()
  
  # for debug
  # print(npy_result)
  
  npy_result = npy_result / 2.5
  #
  # # # Create Image
  dst_img = createResultImg( src_img, npy_result, npy_label, raibowColor=True )
  dst_img = createOverwrappedImage( src_img, dst_img, doGSConv=True )
  
  if SHOW_IMG:
    cv2.imshow( "Test", dst_img )
    cv2.waitKey( 0 )
  return dst_img


def makeAnswerImg( mask_img, answer_img, thresh_low=0.20, thresh_high=0.80 ):
  assert (len( answer_img.shape ) == 3), "answer_img must be 3-ch color image."
  assert (thresh_low < thresh_high), "thresh_low must be lower than thresh_high"

  npy_label = labeling_from_mask( mask_img )
  R = np.array( [0, 0, 255], dtype=np.uint8 )
  
  npy_result = []
  
  for (labelNum, labels) in enumerate( npy_label ):
    n_of_dmg = 0
    for p in labels:
      if (answer_img[p[0], p[1]] == R).all():
        n_of_dmg += 1
    
    # for Debug
    print( "Label %d: score = %lf" % (labelNum, n_of_dmg / len( labels )) )
    
    # 0.30: Collapsed, 0.60:Half-Collapsed, 0.90:Non-Collapsed
    value = n_of_dmg / len( labels )
    if (value < thresh_low):
      # Non-Collapsed
      npy_result.append( 0.90 )
    elif (thresh_low <= value <= thresh_high):
      # Half-Collapsed
      npy_result.append( 0.60 )
    else:
      # Collapsed
      npy_result.append( 0.30 )
  
  npy_result = np.array( npy_result, dtype=np.float32 )
  dst_img = createResultImg( mask_img, npy_result, npy_label, raibowColor=True )
  return dst_img, npy_result


if __name__ == '__main__':
  target_name = "expr_2"
  
  src_img = cv2.imread( f"img/resource/{target_name}_aerial.png", cv2.IMREAD_COLOR )
  mask_img = cv2.imread( f"img/resource/{target_name}_mask.png", cv2.IMREAD_GRAYSCALE )
  answer_img = cv2.imread( f"img/resource/{target_name}_answer.png", cv2.IMREAD_COLOR )
  
  dst_img, npy_result = makeAnswerImg( mask_img, answer_img )
  cv2.imshow( "Test", dst_img )
  cv2.waitKey( 0 )
  np.save( f"data/answer_{target_name}.npy", npy_result )
  print( "Saved Answer Result !! - " + f"data/answer_{target_name}.npy" )
  dst_img = createOverwrappedImage( src_img, dst_img, doGSConv=True )
  cv2.imshow( "Test", dst_img )
  cv2.waitKey( 0 )
  cv2.imwrite( f"img/result/answer_{target_name}.png", dst_img )
  
  # for i in ["1", "2", "3"]:
  # for i in ["1"]:
  #   for sel in ['roi', 'blur_roi']:
  #
  #     for feat in ['length', 'endpoints', 'branches', 'passings']:
  #       ret = saveEdgeFeatureAsImage("img/resource/aerial_" + sel + i + ".png",
  #                   "img/resource/mask_new_roi" + i + ".png",
  #                   "data/new_" + sel + i + "_edge_feat.npy",
  #                                    feat,
  #                                    SHOW_LOG=True, doDataMod=True)
  #
  #       cv2.imwrite("img/result_" + feat + "_" + sel + i + ".png", ret)
  
  # npy_data = np.load('data/final_result_aerial_roi1_customblur.npy')
  # npy_data = np.load('data/npy_roi1_heuristics.npy')
  # src_img = cv2.imread('img/resource/aerial_roi1_customblur.png', cv2.IMREAD_COLOR)
  # mask_img = cv2.imread('img/resource/mask_roi1.png', cv2.IMREAD_GRAYSCALE)
  # npy_label = ml.getMaskLabel(mask_img)
  
  # npy_data = np.append(np.array([0.0]), npy_data)
  #
  # npy_data = npy_data * 0.30
  
  # dst_img = createResultImg(src_img, npy_data, npy_label, raibowColor=True)
  # dst_img = createOverwrappedImage(src_img, dst_img, )
  #
  # cv2.imshow("Test", dst_img)
  # cv2.imwrite("img/result/roi1_heuristics.png", dst_img)
  # cv2.waitKey(0)
