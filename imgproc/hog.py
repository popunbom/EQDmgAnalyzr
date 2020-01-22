# /usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: popunbom <fantom0779@gmail.com>
# Created At: 2019/10/10

"""
imgproc/hog.py : HoG(Histogram of Gradient) 特徴量によるテクスチャ解析
"""

import numpy as np
from skimage.feature import hog

from utils.assertion import TYPE_ASSERT
from utils.logger import ImageLogger


class HoGFeature:
    """
    Attributes
    ----------
    src_img : numpy.ndarray
        入力画像(グレースケール)
    logger : ImageLogger
        ImageLogger インスタンス
        指定された場合、処理結果画像を出力する
    options : dict
        skimage.feature.hog に渡されるオプション
        calc_features で渡される options によって
        上書きされることはない
    """
    
    def __init__( self, src_img, logger ) -> None:
        """
        
        Parameters
        ----------
        src_img : numpy.ndarray
            入力画像
        logger : ImageLogger
            ImageLogger インスタンス
            指定された場合、処理結果画像を出力する
        """
        super().__init__()
        
        TYPE_ASSERT( src_img, np.ndarray )
        TYPE_ASSERT( logger, [None, ImageLogger] )
        
        self.logger = logger
        
        self.options = dict(
            visualize=True,
            feature_vector=False,
            multichannel=False
        )
        
        self.hog_image = None
        self.features = None
        
        self.src_img = src_img
        if src_img.ndim == 3:
            self.options.update(
                multichannel=True
            )
    
    
    def calc_features( self, **options ):
        """
        特徴量計算を行う
        
        Parameters
        ----------
        options : dict
            HoG 計算のオプション
            指定しない場合、デフォルト値が使用される
            詳細は skimage.feature.hog を参照すること

        Returns
        -------
        list of numpy.ndarray
            特徴量, HoG画像
            
        Notes
        -----
        feature_vector が False の場合、特徴量行列は
        M x N x C x D x O の5次元行列になる
            M: img.shape[0] / pixels_per_cell[0]
            N: img.shape[1] / pixels_per_cell[1]
            C, D: cells_per_block
            O: orientations
        feature_vector が True の場合、上記行列が
        1次元に flatten される
        """
        
        self.options = dict(
            list( options.items() ) + list( self.options.items() )
        )
        
        self.features, self.hog_image = hog(
            self.src_img,
            **self.options
        )
        
        if self.logger:
            self.logger.logging_dict( self.options, "options" )
            self.logger.logging_ndarray( self.features, "features" )
            self.logger.logging_img( self.hog_image, "hog_image" )
        
        return self.features, self.hog_image
    
    
    def get_statistics_per_block( self, stat="variance" ):
        # TODO: 統計量はどうやって実装する？
        pass
