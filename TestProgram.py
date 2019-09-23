import cv2
from matplotlib import pyplot as plt
import numpy as np


def on_change( val ):
    th = cv2.getTrackbarPos( 'thresh', 'image' )
    cv2.imshow( 'image', (img > th).astype( np.uint8 ) * 255 )

img = cv2.imread( "img/Lenna.png", cv2.IMREAD_GRAYSCALE )
cv2.namedWindow( 'image' )

# cv2.createTrackbar( 'thresh', 'image', 0, 255, on_change )

# plt.figure()
plt.imshow(img)
plt.show()

# cv2.imshow( 'image', img)
# cv2.waitKey(0)

# while (1):
#     k = cv2.waitKey( 1 ) & 0xFF
#     if k == 27:
#         break
