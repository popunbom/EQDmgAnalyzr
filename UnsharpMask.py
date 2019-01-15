#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy.signal as sig
import cv2
import numpy as np

##### KERNEL 3x3#####
#  1/9, 1/9, 1/9
#  1/9, 1/9, 1/9
#  1/9, 1/9, 1/9
##### KERNEL 3x3#####



# def unsharpMask(img):
#   assert 2 <= img.ndim <= 3, "'img' must be 2d array"
#   KERN = np.array( [ [1/25] * 5 ] * 5, dtype=np.float64 )
#
#   if img.ndim == 2:
#     convolved = sig.convolve2d(img, KERN)
#   elif img.ndim == 3:
#     convolved = np.zeros( img.shape, dtype=img.dtype )
#     for ch in range(img.shape[2]):
#       convolved[:,:,ch] = sig.convolve2d( img[:,:,ch], KERN, mode='same' )
#
#   convolved = convolved.astype(img.dtype)
#   return convolved

def unsharpMask(img, k=1.0):
  X = [
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
  ]
  Y = [
    [-k/9,  -k/9, -k/9],
    [-k/9, 8*k/9, -k/9],
    [-k/9,  -k/9, -k/9]
  ]
  kern = np.array(X) + np.array(Y)
  
  print(kern)
  print(np.sum(kern))
  
  return cv2.filter2D(img, cv2.CV_8U, kern)

if __name__ == '__main__':
    img = cv2.imread("img/test_image_fix.png", cv2.IMREAD_COLOR)
    
    result = unsharpMask(img, k=8)
    
    cv2.imwrite("img/test_image_fix_unsharp.png", result)