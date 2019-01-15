import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure
import os


ORIENTATIONS = 24
PIXELS_PER_CELL = (15, 15)
CELLS_PER_BLOCK = (5, 5)


# img_path = "img/resource/label.bmp"
# img_path = "img/resource/label_with_label.png"
img_path = "img/resource/aerial_roi2.png"
# img_path = "/Users/popunbom/Google Drive/IDE_Projects/--Resources--/IMG_6955-qv.jpg"
# img_path = "/Users/popunbom/Google Drive/IDE_Projects/--Resources--/IMG_6942-qv.jpg"

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


fd, hog_image = hog(img, orientations=ORIENTATIONS, pixels_per_cell=PIXELS_PER_CELL, cells_per_block=CELLS_PER_BLOCK, visualise=True, feature_vector=False)


# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(hog_image.min(), hog_image.max()), out_range=(0, 255))


hog_image_rescaled = (hog_image / hog_image.max() * 255.0).astype(np.uint8)

# Calc Block Variance
fd_block_var = np.array([[np.var(np.array([ np.sum(fd[i, j, :, :, k]) for k in range(0, ORIENTATIONS) ])) for j in range(fd.shape[1])] for i in range(fd.shape[0])])
fd_block_var = (fd_block_var / fd_block_var.max() * 255.0).astype(np.uint8)


# cv2.imshow("HoG", hog_image_rescaled)
# cv2.waitKey(0)

target_name = os.path.splitext(os.path.basename(img_path))[0]
print(target_name)

cv2.imwrite("img/result_{}_HoG_label_OR{}_PPC{}_CPB{}.png".format(target_name, ORIENTATIONS, PIXELS_PER_CELL[0], CELLS_PER_BLOCK[0]), hog_image_rescaled)
cv2.imwrite("img/result_{}_HoG_BlockVariance_label_OR{}_PPC{}_CPB{}.png".format(target_name, ORIENTATIONS, PIXELS_PER_CELL[0], CELLS_PER_BLOCK[0]), fd_block_var)
