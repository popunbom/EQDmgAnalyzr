# -*- coding: utf-8 -*-
# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2019-05-22

# This is a part of EQDmgAnalyzr

# -*- coding: utf-8 -*-
# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2019-05-22

# This is a part of EQDmgAnalyzr

import cv2
import numpy as np

from ParamAdjust import ParamAdjust


class CannyParamAdjust( ParamAdjust ):
  
  
  def update( self, *args ):
    th_1, th_2 = [x.get_val() for x in self.get_trackbar( 'th_1', 'th_2' )]
    subimg_src, subimg_dst = self.get_subimage( 'src', 'dst' )
    
    gray = cv2.cvtColor( subimg_src.img, cv2.COLOR_BGR2GRAY )
    subimg_dst.img = cv2.Canny( gray, th_1, th_2 )
    
    print( "Updated" )


if __name__ == '__main__':
  IMG_PATH = "/Users/popunbom/Downloads/IMG_6955-qv_shadow_with_transparented.png"
  
  img = cv2.imread( IMG_PATH, cv2.IMREAD_COLOR )
  
  TRACKBARS = {
    "th_1": {
      "label"    : "th_1 (0, 255, 1)",
      "val_range": (0, 256)
    },
    "th_2": {
      "label"    : "th_2 (0, 255, 1)",
      "val_range": (0, 256)
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
  
  inst = CannyParamAdjust( TRACKBARS, IMAGES, name_trackbar_window="Trackbars" )
  inst.run()
