import cv2
import os
import numpy as np

import CommonProcedures as cp

# ROOT_DIR = 'img/divided/aerial_roi1'
#
# for img_path in sorted([ x for x in os.listdir(ROOT_DIR) if 'canny' not in x]):

# img = cv2.imread( os.path.join(ROOT_DIR, img_path), cv2.IMREAD_GRAYSCALE)

IMG_PATH = "/Users/popunbom/Google Drive/.research_resources/img_resource/cropped.png"
# IMG_PATH = "/Users/popunbom/Google Drive/.research_resources/img_resource/IMG_6943-qv.jpg"

img = cv2.imread( IMG_PATH, cv2.IMREAD_GRAYSCALE )

lsd = cv2.createLineSegmentDetector( _refine=cv2.LSD_REFINE_STD )

lines, width, prec, nfa = lsd.detect( img )

drawn_img = lsd.drawSegments( img, lines )

# cp.imshow( drawn_img, title="LSD (refine = cv2.LSD_REFINE_STD)" )
cv2.imshow( "LSD (refine = cv2.LSD_REFINE_STD)", drawn_img )
cv2.waitKey(0)
