# -*- coding: utf-8 -*-
# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2019-03-10

# This is a part of EQDmgAnalyzr

import cv2
import numpy as np


img = cv2.imread("img/resource/aerial_roi1.png", cv2.IMREAD_COLOR)
gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY ).astype(np.float32)

dst = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
  cv2.destroyAllWindows()