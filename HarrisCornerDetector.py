# -*- coding: utf-8 -*-
# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2019-05-22

# This is a part of EQDmgAnalyzr

import cv2
import numpy as np


from ParamAdjust import ParamAdjust


class HarrisParamAdjust( ParamAdjust ):
  
  
  def update( self, *args ):
    block_size, k_size, k = [x.get_val() for x in self.get_trackbar( 'block_size', 'k_size', 'k' )]
    subimg_src, subimg_dst = self.get_subimage( 'src', 'dst' )
    
    gray = cv2.cvtColor( subimg_src.img, cv2.COLOR_BGR2GRAY ).astype( np.float32 )
    dst = cv2.cornerHarris( gray, (block_size + 1), (2 * k_size + 1), (k / 100) )
    
    dst = cv2.dilate( dst, None )
    
    result = np.copy( subimg_src.img )
    
    result[dst > 0.01 * dst.max()] = [0, 0, 255]
    
    subimg_dst.img = result
    
    print( "Updated" )


if __name__ == '__main__':
  img = cv2.imread( "img/resource/aerial_roi1.png", cv2.IMREAD_COLOR )
  
  TRACKBARS = {
    "block_size": {
      "label"    : "Param: blockSize (1, 10)",
      "val_range": (0, 8)
    },
    "k_size"    : {
      "label"    : "Param: kSize ( 2*n + 1)",
      "val_range": (0, 10)
    },
    "k"         : {
      "label"    : "Param: k ( 10^-2 )",
      "val_range": (0, 100)
    }
  }
  
  IMAGES = {
    "src": {
      "label": "Source",
      "img"  : img
    },
    "dst": {
      "label": "Result",
      "img"  : None
    }
  }
  
  inst = HarrisParamAdjust( TRACKBARS, IMAGES, name_trackbar_window="Trackbars" )
  inst.run()
