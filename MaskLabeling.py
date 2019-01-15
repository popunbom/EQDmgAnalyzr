import random
import copy
import numpy as np
import cv2

FILE_NAME = "mask_invert.png"


def getMaskLabel( img_mask ):
  assert (len( img_mask.shape ) == 2), "input image must be 1-ch"
  
  _, mask_bin = cv2.threshold( img_mask, 0, 255, cv2.THRESH_BINARY )
  
  print( "Labeling... ", end="", flush=True )
  nLabels, labels = cv2.connectedComponents( mask_bin )
  print( "done! (Labels = %d)" % nLabels, flush=True )
  
  print( "Create Label Array ... ", end="", flush=True )
  lbl = []
  for i in range( nLabels ):
    lbl.append( [(-1, -1)] )
  
  for y in range( 0, img_mask.shape[0] ):
    for x in range( 0, img_mask.shape[1] ):
      lbl[labels[y, x]].append( (y, x) )
      
  
  np_lbl = np.array( lbl )
  
  # for debug
  for (i, x) in enumerate(np_lbl):
    if ( len(x) < 5):
      print("Label %d: S = %d" % (i, len(x)))
      print(x)
  
  print( "done! ", flush=True )
  return np_lbl


if __name__ == '__main__':
  img_mask = cv2.imread( str( "img/" + FILE_NAME ), cv2.IMREAD_GRAYSCALE )
  
  np_lbl = getMaskLabel( img_mask )
  save_name = str( "data/label_" + FILE_NAME + ".npy" )
  
  np.save( save_name, np_lbl )
  print( "done! saved as \"./%s\"" % save_name, flush=True )
