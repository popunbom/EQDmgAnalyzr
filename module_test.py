#/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: popunbom <fantom0779@gmail.com>
# Created At: 2019/09/23

import cv2
from utils.mpl import show_images
from utils.logger import ImageLogger
from os import path

PATH_ROOT = "./img/resource"

img_names = [
    "aerial_blur_roi1.png",
    "aerial_blur_roi2.png",
    "aerial_blur_roi3.png",
    "aerial_roi4.png"
]


img_list = [
    cv2.imread(
        path.join(PATH_ROOT, name),
        cv2.IMREAD_GRAYSCALE
    )
    for name in img_names
]

for img, name in zip(img_list, img_names):
    if img is None:
        print(f"Image cannot be loaded: {name}")
        
logger = ImageLogger("./img/tmp/module_test")

show_images(
    list_img=img_list,
    plt_title="Images",
    list_title=[
        "ROI 1 (Blurred)",
        "ROI 2 (Blurred)",
        "ROI 3 (Blurred)",
        "ROI 4"
    ],
    list_cmap=[
        "gray", "jet", "jet", "gray"
    ],
    tuple_shape=(2, 3),
    logger=logger
)
