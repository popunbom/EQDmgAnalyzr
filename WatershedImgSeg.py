#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# REF: https://qiita.com/ysdyt/items/5972c9520acf6a094d90

import os
import cv2
import numpy as np

from utils.mpl import show_images


def watershed(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh, img_bin = cv2.threshold(img_gs, 0, 255, cv2.THRESH_OTSU)
    if np.sum(img_bin == 0) > np.sum(img_bin == 255):
        img_bin = cv2.bitwise_not(img_bin)


    print(f"Threshold: {thresh}")

    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=np.uint8)

    img_morph = img_bin
    img_morph = cv2.morphologyEx(
        img_morph, cv2.MORPH_CLOSE, kernel, iterations=1)
    img_morph = cv2.morphologyEx(
        img_morph, cv2.MORPH_OPEN, kernel, iterations=1)

    sure_bg = img_morph

    dist_transform = cv2.distanceTransform(img_morph, cv2.DIST_L2, 5)

    thresh, sure_fg = cv2.threshold(
        dist_transform, 0.5 * dist_transform.max(), 255, 0)

    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)

    show_images([img_gs, img_bin, img_morph, dist_transform, sure_fg, unknown],
               list_title=["Grayscale", "Binary", "Morph",
                           "Distance", "Foreground", "Unknown"],
               plt_title=os.path.basename(img_path))
    ret, markers = cv2.connectedComponents(sure_fg)
    markers += 1

    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    show_images([img, img_bin, markers],
               list_title=["Input", "Binary", "Watershed"],
               list_cmap=['gray', 'rainbow'],
               tuple_shape=(1, 3),
               plt_title=os.path.basename(img_path)
               )


if __name__ == '__main__':
    for i in range(100):
        watershed(f"img/divided/aerial_roi1/{i:05d}.png")
    # watershed("/Users/popunbom/Google Drive/IDE_Projects/PyCharm/DmgAnalyzr/img/resource/aerial_blur_roi1.png")
