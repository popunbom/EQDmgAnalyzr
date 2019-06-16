#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

img = cv2.imread("img/Lenna.png")

lsd_result = cv2.LineSegmentDetector.detect(img)
