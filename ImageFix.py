#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np

PATH_IMG = "img/Lenna.png"
PADDING_WIDTH = 1
PADDING_VALUE = 0

src = cv2.imread(PATH_IMG, cv2.IMREAD_COLOR)
base_name = os.path.splitext(os.path.basename(PATH_IMG))[0]
assert src.ndim == 3, "'src' must be 3-ch color image."

moved = dict()
for h in ['left', 'right']:
  moved[h] = dict()
  for v in ['top', 'bottom']:
    pos = {
      'x': 0 if h == 'left' else src.shape[1],
      'y': 0 if v == 'top'  else src.shape[0],
    }
    
    print( f"moved[{h}][{v}] = np.insert( np.insert( src, [{pos['x']}] * {PADDING_WIDTH}, {PADDING_VALUE}, axis=1 ), [{pos['y']}] * {PADDING_WIDTH}, {PADDING_VALUE}, axis=0 )" )
    
    moved[h][v] = np.insert(
                    np.insert(
                      src,
                      [pos['x']] * PADDING_WIDTH,
                      PADDING_VALUE,
                      axis=1 ),
                    [pos['y']] * PADDING_WIDTH,
                    PADDING_VALUE,
                    axis=0 )
    
for h in ['left', 'right']:
  for v in ['top', 'bottom']:
    cv2.imwrite(f"img/{base_name}_{h}_{v}.png", moved[h][v])
    
result = np.zeros( moved['left']['top'].shape, dtype=np.float64 )
for h in ['left', 'right']:
  for v in ['top', 'bottom']:
    result = result + moved[h][v]
    
result = (result / 4.0).astype(np.uint8)

cv2.imwrite(f"img/{base_name}_merge_average.png", result)