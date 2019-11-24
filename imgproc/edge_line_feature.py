#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2019-11-24
# This is a part of EQDmgAnalyzr

import cv2
import numpy as np
from tqdm import trange

from imgproc.utils import get_window_rect
from utils.assertion import TYPE_ASSERT, NDARRAY_ASSERT, SAME_SHAPE_ASSERT
from utils.common import eprint as _eprint
from utils.logger import ImageLogger

DEBUG = False


def eprint( *args, **kwargs ):
    kwargs.update( debug_flag=DEBUG )
    _eprint( *args, **kwargs )


class EdgeLineFeatures:
    BASE_TEMPLATES = {
        "endpoint": [
            np.array( [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]
            ], dtype=np.int8 ),
            np.array( [
                [0, 0, 0],
                [0, 1, 0],
                [-1, 1, -1]
            ], dtype=np.int8 ),
            np.array( [
                [0, 0, 0],
                [0, 1, 0],
                [1, 0, 0]
            ], dtype=np.int8 ),
        ],
        "branch"  : [
            np.array( [
                [-1, 1, -1],
                [1, 1, 1],
                [0, 0, 0]
            ], dtype=np.int8 ),
            np.array( [
                [1, 0, 0],
                [0, 1, 0],
                [1, 0, 1]
            ], dtype=np.int8 ),
            np.array( [
                [-1, 1, -1],
                [1, 1, 1],
                [-1, 1, -1]
            ], dtype=np.int8 ),
            np.array( [
                [1, 0, 1],
                [0, 1, 0],
                [1, 0, 1]
            ], dtype=np.int8 )
        ]
    }
    
    LABELS = {
        "UNDEFINED": 255,
        "BG"       : 0,
        "endpoint" : 1,
        "branch"   : 2,
        "passing" : 3,
    }
    
    DONT_CARE = -1
    K_SIZE = 3
    
    def __init__( self, img, logger=None ):
        """
        コンストラクタ
        
        Parameters
        ----------
        img : numpy.ndarray
            入力画像
            - グレースケール画像
        logger : ImageLogger, defaults None
            処理途中の画像をロギングする ImageLogger
        """
        super().__init__()
        
        NDARRAY_ASSERT( img, ndim=2 )
        
        self.BG, self.FG = -1, -1
        self.img = img.copy()
        
        if self.img.dtype != np.uint8:
            self.img = (img / img.max()).astype( np.uint8 )
        
        if np.unique( self.img ).size == 2:
            self.BG, self.FG = np.unique( self.img )
        
        self.classified = np.full(
            img.shape[:2],
            fill_value=self.LABELS["UNDEFINED"],
            dtype=np.uint8
        )
        
        self.metrics = {
            "total_length"  : 0,
            "average_length": 0,
        }
        
        # Generate Rotated template
        self.TEMPLATE = {
            template_name: sum( [
                [
                    np.rot90( template, n_rotate )
                    for n_rotate in range( 4 )
                ]
                for template in templates
            ], [] )
            for template_name, templates in self.BASE_TEMPLATES.items()
        }
        
        self.logger = logger
    
    def check_image( self ):
        img = self.img
        
        assert np.unique( img ).size == 2, \
            "'self.img' must be binarized"
    
    def match_to_template( self, img_roi, template ):
        """
        切り出した領域がテンプレートにマッチするかどうか調べる
        Parameters
        ----------
        img_roi : numpy.ndarray
            切り出した領域
        template : numpy.ndarray
            テンプレート

        Returns
        -------
        bool
            切り出した領域がテンプレートにマッチしたかどうか

        Notes
        -----
        `img_roi` と `template` は同じ大きさである必要がある
        """
        
        DONT_CARE = self.DONT_CARE
        
        SAME_SHAPE_ASSERT( img_roi, template )
        
        return np.array_equal(
            img_roi[template != DONT_CARE],
            template[template != DONT_CARE]
        )
    
    def do_canny( self, thresh_1, thresh_2 ):
        """
        Canny のエッジ検出を行う
        
        Parameters
        ----------
        thresh_1, thresh_2 : int
            閾値

        """
        logger = self.logger
        self.img = cv2.Canny(
            self.img,
            thresh_1,
            thresh_2
        ).astype( bool ).astype( bool )
        
        if logger:
            logger.logging_img( self.img, "canny" )
            logger.logging_dict( {
                "thresh_1": thresh_1,
                "thresh_2": thresh_2,
            }, "params_canny" )
        
        self.BG, self.FG = 0, 1
    
    def get_as_image( self ):
        """
        エッジ画素分類結果を画像で可視化する
        
        Returns
        -------
        numpy.ndarray
            可視化したエッジ画素分類結果画像
        """
        LABELS = self.LABELS
        classified = self.classified
        
        COLORS = {
            # [B, G, R] の順番
            "passing" : [255, 128, 128],
            "branch" : [0, 0, 255],
            "endpoint": [0, 255, 255]
        }
        
        img = np.zeros(
            (*classified.shape[:2], 3),
            dtype=np.uint8
        )
        
        for type, color in COLORS.items():
            img[classified == LABELS[type]] = color
            
        return img
        
    def classify_pixel( self ):
        """
        エッジ画素分類を行う
        
        - テンプレートマッチングにより
          「端点」「分岐点」を探索
        
        - テンプレートにマッチしなかったエッジ画素は
          「通過点」とする

        Returns
        -------
        numpy.ndarray
            エッジ画素分類結果
        """
        self.check_image()
        
        BG, FG = self.BG, self.FG
        LABELS = self.LABELS
        TEMPLATE = self.TEMPLATE
        k_size = self.K_SIZE
        logger = self.logger
        img = self.img
        classified = self.classified
        height, width = self.img.shape[:2]
        
        # 全画素ループ (tqdm で進捗可視化)
        for i in trange( height, desc="Height", leave=False ):
            for j in trange( width, desc="Width", leave=False ):
                
                # 背景画素だったらスキップ
                if (img[i, j] == BG):
                    classified[i, j] = LABELS["BG"]
                    continue
                
                # img_roi: (i, j) を中心に k_size の大きさの矩形
                img_roi = img[
                    get_window_rect(
                        img_shape=self.img.shape,
                        center=(i, j),
                        wnd_size=k_size,
                        ret_type="slice"
                    )
                ].copy()
                img_roi[img_roi == FG] = 1
                
                # 「端点」「分岐点」へのマッチング
                for type in TEMPLATE.keys():
                    for template in TEMPLATE[type]:
                        
                        # もしテンプレートにマッチングしたら
                        if self.match_to_template( img_roi, template ):
                            
                            # classified に結果を保持
                            classified[i, j] = LABELS[type]
                            
                            eprint(
                                f"Edge Classified: ({i}, {j}) as '{type}'",
                                use_tqdm=True
                            )
        
        # テンプレートにマッチしなかったエッジ画素 → 「通過点」
        classified[classified == LABELS["UNDEFINED"]] = LABELS["passing"]
        
        if logger:
            logger.logging_img(classified, "classified")
            logger.logging_img(self.get_as_image(), "classified_visualized")
        
        return classified
    
    def calc_metrics( self ):
        """
        統計量の計算を行う
        
        - `total_length` : エッジ画素の総数
        
        - `average_length` : 平均エッジ長
        
        Returns
        -------
        dict
            統計量
        """
        
        FG = self.FG
        logger = self.logger
        img = self.img
        classified = self.classified
        metrics = self.metrics
        
        endpoint, passing = self.LABELS["endpoint"], self.LABELS["passing"]
        
        # total_length : エッジ画素の総数
        metrics["total_length"] = img[img == FG].size
        
        # average_length : 平均エッジ長
        # - 分岐点を除去してラベリング
        # - 平均エッジ帳 = (「端点」「通過点」の総数) / (ラベル数)
        
        extracted = (classified == endpoint) | (classified == passing)
        
        n_labels, labels = cv2.connectedComponents(
            (extracted * 255).astype( np.uint8 ),
            connectivity=8
        )
        
        metrics["average_length"] = np.count_nonzero( extracted ) / n_labels
        
        if logger:
            logger.logging_dict(metrics, "metrics")
            logger.logging_img(labels, "edge_lines", cmap="jet")

        return metrics
