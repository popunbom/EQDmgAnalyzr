import cv2
import numpy as np
import os

N_OF_ANGLES = 8
BLOCK_SIZE = 75

# img_path = "img/resource/label.bmp"
# img_path = "img/resource/label_with_label.png"
img_path = "img/resource/aerial_roi1.png"

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = img.astype( np.float64 ) / 255.0

dy, dx = np.gradient( img )

G = np.sqrt( dx ** 2 + dy ** 2 )
G /= G.max()

A = np.arctan2( dy, dx )

A_as_degrees = np.degrees((A + abs(A.min())))

A_quantized = (A_as_degrees / (360 / N_OF_ANGLES)).astype( np.int16 )

A = A_quantized

result = np.zeros( tuple( ((s - BLOCK_SIZE) + 1) for s in img.shape ), dtype=np.float64 )

# print("Create A_rois ... ", end = "", flush=True)
# A_roiss = np.array([ np.array([ np.ravel(A[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]) for x in range( 0, result.shape[1] )], dtype=np.int16) for y in range( 0, result.shape[0] ) ], dtype=np.int16)
# print("done!")
# print("Create G_rois ... ", end = "", flush=True)
# G_roiss = np.array([ np.array([ np.ravel(G[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]) for x in range( 0, result.shape[1] )], dtype=np.float64) for y in range( 0, result.shape[0] ) ], dtype=np.float64)
# print("done!")
#
# print("Create result ... ", end = "", flush=True)
# result = np.array([ np.array([ np.array([ G_roi[A_roi == d].sum() for d in range(N_OF_ANGLES)]).var() for A_roi, G_roi in zip(A_rois, G_rois) ]) for A_rois, G_rois in zip (A_roiss, G_roiss) ])
# print("done!")

# print(result)


for y in range( 0, result.shape[0] ):
  for x in range( 0, result.shape[1] ):
    # print( "({}, {})".format( x, y ) )
    A_roi = A[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]
    G_roi = G[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]

    if (y % 50 == 0 and x == 0):
      print( y, end=", ", flush=True )

    hist = np.zeros( N_OF_ANGLES, dtype=np.float64 )
    for i, v in zip( np.ravel( A_roi ), np.ravel( G_roi ) ):
      hist[int( i )] += v



    result[int(y), int(x)] = hist.var()

result = (result / result.max() * 255.0).astype( np.uint8 )

target_name = os.path.splitext(os.path.basename(img_path))[0]

cv2.imwrite( "img/result_{}_EdgeVariance_aerial_roi1_BS{}.png".format( target_name, BLOCK_SIZE ), result )
