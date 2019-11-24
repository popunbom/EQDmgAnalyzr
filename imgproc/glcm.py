# -*- coding: utf-8 -*-
# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2019-10-05

# This is a part of EQDmgAnalyzr
"""
imgproc/glcm.py : 同時正規行列(Grey-Level Co-occurrence Matrix) によるテクスチャ解析
"""

from itertools import product

import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from imgproc.utils import get_rect
from utils.common import eprint
from utils.logger import ImageLogger
from utils.assertion import TYPE_ASSERT


# TODO: ウィンドウ処理を追加
class GLCMFeatures:
    """
    Attributes
    ----------
    src_img : numpy.ndarray
        入力画像(グレースケール)
    distances : list of int
        2画素間の距離
        距離ごとに特徴量を計算する
    degrees : list of int
        2画素間の角度 [度]
        角度ごとに特徴量を計算する
    labels : numpy.ndarray
        imgproc/labeling.py によって作成されたラベリングデータ
        - calc_features で use_labels が True の場合、ラベリング
          結果ごとに特徴量を求める
    logger : ImageLogger
        ImageLogger インスタンス
        指定された場合、処理結果画像を出力する
    """
    
    def __init__( self,
                  src_img,
                  distances=[1],
                  degrees=[45],
                  labels=None,
                  logger=None ) -> None:
        """
        Parameters
        ----------
        src_img : numpy.ndarray
            入力画像
            RGB カラー画像が与えられた場合、
            グレースケールに変換される
        distances : list of int
            2画素間の距離
            距離ごとに特徴量を計算する
        degrees : list of int
            2画素間の角度 [度]
            角度ごとに特徴量を計算する
        labels : numpy.ndarray
            imgproc/labeling.py によって作成されたラベリングデータ
            - calc_features で use_labels が True の場合、ラベリング
              結果ごとに特徴量を求める
        logger : ImageLogger
            ImageLogger インスタンス
            指定された場合、処理結果画像を出力する
        """
        super().__init__()
        
        TYPE_ASSERT( src_img, np.ndarray )
        TYPE_ASSERT( distances, list )
        TYPE_ASSERT( degrees, list )
        TYPE_ASSERT( labels, [None, np.ndarray] )
        TYPE_ASSERT( logger, [None, ImageLogger] )
        
        self.distances = distances
        self.degrees = degrees
        self.logger = logger
        self.labels = labels
        
        if src_img.ndim == 3:
            self.src_img = cv2.cvtColor( src_img, cv2.COLOR_BGR2GRAY )

        if self.logger:
            logger.logging_dict(
                {
                    "distances": distances,
                    "degrees"  : degrees
                },
                "options"
            )
    
    def calc_features( self, feature_names, use_labels=False ):
        """
        特徴量計算を行う
        
        `feature_names` には以下の値が使用可能
            - `contrast` : コントラスト
            - `dissimilarity` : 異質度
            - `homogeneity` : 均質性(逆差分モーメント)
            - `ASM` : 二次角度モーメント (anguler second moment)
            - `energy` : エネルギー
            - `correlation` : 相関
            
        Parameters
        ----------
        feature_names : list of str
            取得する特徴量
        use_labels : bool
            ラベリングを行うかどうか
            
            `True` の場合, `self.labels` がセットされている
            必要がある
        
        Returns
        -------
        dict
            特徴量データ
        """
        TYPE_ASSERT( feature_names, (list, str) )
        TYPE_ASSERT( use_labels, bool )
        
        keys_and_indices = list(
            product( enumerate( self.distances ), enumerate( self.degrees ) )
        )
        
        if isinstance(feature_names, str):
            feature_names = [ feature_names ]
        
        def _calc_features( _img, _feature_names ):
            eprint( "Calculating GLCM [ image.shape = {shape} ] ... ".format(
                shape=_img.shape ) )
            
            _glcm = greycomatrix( _img, self.distances, self.degrees )
            
            return {
                _feature_name: {
                    (_dist, _deg): greycoprops( _glcm, _feature_name )[_dist_idx][_deg_idx]
                    for (_dist_idx, _dist), (_deg_idx, _deg) in keys_and_indices
                }
                for _feature_name in _feature_names
            }
        
        if use_labels:
            assert self.labels is not None, "Labeling data is not set."

            features = list()

            for points in self.labels:
                (yMin, xMin), (yMax, xMax) = get_rect( self.src_img.shape,
                                                       points )
                roi = self.src_img[yMin:yMax, xMin:xMax]
    
                features.append( _calc_features( roi, feature_names ) )
        
        else:
            features = _calc_features( self.src_img, feature_names )
        
        if self.logger:
            
            if use_labels:
                features = {
                    "label_{i}".format( i=i ): {
                        feature_name: { str( k ): v
                                        for k, v in values.items() }
                        for feature_name, values in feature.items()
                    }
                    for i, feature in enumerate( features )
                }
            
            else:
                features = {
                    feature_name: { str( k ): v
                                    for k, v in values.items() }
                    for feature_name, values in features.items()
                }
            
            self.logger.logging_dict( features, "features", overwrite=True )
        
        return features
