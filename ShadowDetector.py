# -*- coding: utf-8 -*-
# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2019-05-20

# This is a part of EQDmgAnalyzr

# This is the implementation of algorithm in following articles
#   "Shadow Detection and Removal from a Single Image Using LAB Color Space"

import cv2
import numpy as np

def shadow_detector(img_rgb, threshold=256):
  assert (type(img_rgb) == np.ndarray and len(img_rgb.shape) == 3) or type(img_rgb) == str, \
    "arguments 'img_rgb' must be str of path to img or np.ndarray RGB color image data"
  
  if type(img_rgb) == str:
    img_rgb = cv2.imread(img_rgb, cv2.IMREAD_COLOR)
  
  # Convert the RGB image to a LAB image.
  img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2LAB)
  L, A, B = [ img_lab[:,:,i] for i in range(3) ]
  
  shadow_mask = np.zeros( img_rgb.shape[:2], dtype=np.bool )
  
  # Compute the mean values of the pixels in L, A and B planes of the image
  # separately.
  mean = {
    'L': np.mean(L),
    'A': np.mean(A),
    'B': np.mean(B)
  }
  
  # If mean (A) + mean (B) ≤ 256
  if mean['A'] + mean['B'] <= threshold:
    # Classify the pixels with a value in L ≤(mean(L) – standard deviation (L)/3) as shadow pixels and others as non-shadow pixels.
    shadow_mask[ L <= (mean['L'] + np.std(L) / 3) ] = True
  else:
    # Else classify the pixels with lower values in both L and B planes as shadow pixels and others as non-shadow pixels.
    shadow_mask[ np.logical_and(L < mean['L'], B < mean['B']) ] = True
    
  img_shadow_mask = shadow_mask.astype(np.uint8) * 255
  
  cv2.imshow("img_shadow_mask", img_shadow_mask)
  cv2.waitKey()
  
  return img_shadow_mask
    
  
if __name__ == '__main__':
  # IMG_PATH = "/Users/popunbom/Downloads/bell_width_shadow.jpg"
  IMG_PATH = "/Users/popunbom/Downloads/IMG_6955-qv.png"
  shadow_mask = shadow_detector(IMG_PATH, 300)
  
  cv2.imwrite("/Users/popunbom/Downloads/shadow_mask.png", shadow_mask)
