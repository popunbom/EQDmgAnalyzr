# -*- coding: utf-8 -*-
import os
import cv2
import CommonProcedures as cp
import MaskLabeling as ml

import numpy as np

def proc1(path_src_img, path_mask_img, dir_name):
  
  print(" path_src_img = ", path_src_img)
  print("path_mask_img = ", path_mask_img)
  print("     dir_name = ", dir_name)
  
  src_img = cv2.imread(path_src_img, cv2.IMREAD_COLOR)
  mask_img = cv2.imread(path_mask_img, cv2.IMREAD_GRAYSCALE)

  npy_label = ml.getMaskLabel(mask_img)

  if (not os.path.exists("img/divided/" + dir_name)):
    os.mkdir("img/divided/" + dir_name)

  for i in range(len(npy_label)):
    P = cp.getRect(src_img.shape, npy_label[i])

    cv2.imwrite("img/divided/" + dir_name + "/%05d.png" % i, src_img[P[0]:P[1], P[2]:P[3]])
    cv2.imwrite("img/divided/" + dir_name + "/canny_%05d.png" % i,
                cv2.Canny(src_img[P[0]:P[1], P[2]:P[3]], 126, 174))

def proc2():
  img_src = cv2.imread("img/aerial_roi2.png", cv2.IMREAD_COLOR)
  img_mask = cv2.imread("img/mask_new_roi2_fixed.png", cv2.IMREAD_GRAYSCALE)
  
  dst = cv2.add(img_src, np.zeros(img_src.shape, dtype=np.uint8), mask=img_mask)
  bg_img = np.ones(img_src.shape, dtype=np.uint8)
  bg_img[:,:] = np.array([255, 0, 255])
  
  dst = dst + bg_img
  
  cv2.imshow("Test", dst)
  cv2.waitKey(0)
  cv2.imwrite("img/aerial_clipped_roi2.png", dst)

if __name__ == '__main__':
  
  img = cv2.imread("img/divided/aerial_roi1/00003.png", 0)
  cv2.imshow("Test", img)
  cv2.waitKey(0)
  # # for sel1 in ["1", "2"]:
  # for sel1 in ["3"]:
  #   for sel2 in ["roi", "blur_roi"]:
  #     proc1("img/resource/aerial_"+sel2+sel1+".png",
  #           "img/resource/mask_new_roi"+sel1+".png",
  #           "new_"+sel2+sel1)
  #
  # # proc1("img/aerial_blur_roi2.png", "img/mask_new_roi2.png", "new_blur_roi2")
  # # proc2()