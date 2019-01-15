import numpy as np
import cv2


if __name__ == '__main__':
    img = cv2.imread("img/aerial_only_roi1.png")
    dst = cv2.pyrMeanShiftFiltering(img, 30.0, 30.0)

    cv2.imshow("Test", dst)
    cv2.waitKey(0)