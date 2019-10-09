# -*- coding: utf-8 -*-
# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2019-10-05

# This is a part of EQDmgAnalyzr
"""
imgproc/glcm.py : 同時正規行列(Grey-Level Co-occurrence Matrix) によるテクスチャ解析
"""

from itertools import product

import cv2
from skimage.feature import greycomatrix, greycoprops
from imgproc.utils import get_rect
from utils.common import eprint
from utils.logger import ImageLogger
from utils.assertion import *


class GLCMFeatures:
    def __init__(self,
                 src_img,
                 distances=[1],
                 degrees=[45],
                 labels=None,
                 logger=None) -> None:
        super().__init__()

        TYPE_ASSERT(src_img, np.ndarray)
        TYPE_ASSERT(distances, list)
        TYPE_ASSERT(degrees, list)
        TYPE_ASSERT(labels, [None, np.ndarray])
        TYPE_ASSERT(logger, [None, ImageLogger])

        self.distances = distances
        self.degrees = degrees
        self.logger = logger
        self.labels = labels

        if src_img.ndim == 3:
            self.src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    def calc_features(self, feature_names, use_labels=False):
        TYPE_ASSERT(feature_names, list)
        TYPE_ASSERT(use_labels, bool)

        keys_and_indices = list(
            product(enumerate(self.distances), enumerate(self.degrees)))

        def _calc_features(_img, _feature_names):
            eprint("Calculating GLCM [ image.shape = {shape} ] ... ".format(
                shape=_img.shape))

            _glcm = greycomatrix(_img, self.distances, self.degrees)

            return {
                _feature_name:
                {(_dist, _deg): greycoprops(_glcm,
                                            _feature_name)[_dist_idx][_deg_idx]
                 for (_dist_idx, _dist), (_deg_idx, _deg) in keys_and_indices}
                for _feature_name in _feature_names
            }

        if use_labels:
            assert self.labels is not None, "Labeling data is not set."

            features = list()

            for points in self.labels:
                (yMin, xMin), (yMax, xMax) = get_rect(self.src_img.shape,
                                                      points)
                roi = self.src_img[yMin:yMax, xMin:xMax]

                features.append(_calc_features(roi, feature_names))

        else:
            features = _calc_features(self.src_img, feature_names)

        if self.logger:
            features = {
                "label_{i}".format(i=i): {
                    feature_name: {str(k): v
                                   for k, v in values.items()}
                    for feature_name, values in feature.items()
                }
                for i, feature in enumerate(features)
            }
            self.logger.logging_json(features, "features", overwrite=True)

        return features
