#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2019-11-24
# This is a part of EQDmgAnalyzr
import re
from multiprocessing import current_process, Pool
from time import sleep

import cv2
import numpy as np
from skimage.feature import canny
from tqdm import trange, tqdm

from imgproc.utils import get_window_rect, compute_by_window
from utils.assertion import TYPE_ASSERT, NDARRAY_ASSERT, SAME_SHAPE_ASSERT
from utils.common import eprint as _eprint
from utils.logger import ImageLogger

DEBUG = False


def eprint(*args, **kwargs):
    kwargs.update(debug_flag=DEBUG)
    _eprint(*args, **kwargs)


class EdgeLineFeatures:
    BASE_TEMPLATES = {
        "endpoint": [
            np.array([
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]
            ], dtype=np.int8),
            np.array([
                [0, 0, 0],
                [0, 1, 0],
                [-1, 1, -1]
            ], dtype=np.int8),
            np.array([
                [0, 0, 0],
                [0, 1, 0],
                [1, 0, 0]
            ], dtype=np.int8),
        ],
        "branch"  : [
            np.array([
                [-1, 1, -1],
                [1, 1, 1],
                [0, 0, 0]
            ], dtype=np.int8),
            np.array([
                [1, 0, 0],
                [0, 1, 0],
                [1, 0, 1]
            ], dtype=np.int8),
            np.array([
                [-1, 1, -1],
                [1, 1, 1],
                [-1, 1, -1]
            ], dtype=np.int8),
            np.array([
                [1, 0, 1],
                [0, 1, 0],
                [1, 0, 1]
            ], dtype=np.int8)
        ]
    }
    
    LABELS = {
        "UNDEFINED": 255,
        "BG"       : 0,
        "endpoint" : 1,
        "branch"   : 2,
        "passing"  : 3,
    }
    
    DONT_CARE = -1
    K_SIZE = 3
    
    def __init__(self, img, logger=None):
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
        
        NDARRAY_ASSERT(img, ndim=2)
        
        self.BG, self.FG = -1, -1
        self.img = img.copy()
        
        if self.img.dtype != np.uint8:
            self.img = (img / img.max()).astype(np.uint8)
        
        if np.unique(self.img).size == 2:
            self.BG, self.FG = np.unique(self.img)
        
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
            template_name: sum([
                [
                    np.rot90(template, n_rotate)
                    for n_rotate in range(4)
                ]
                for template in templates
            ], [])
            for template_name, templates in self.BASE_TEMPLATES.items()
        }
        
        self.logger = logger
    
    def check_image(self):
        img = self.img
        
        assert np.unique(img).size == 2, \
            "'self.img' must be binarized"
    
    def match_to_template(self, img_roi, template):
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
        
        SAME_SHAPE_ASSERT(img_roi, template)
        
        return np.array_equal(
            img_roi[template != DONT_CARE],
            template[template != DONT_CARE]
        )
    
    def do_canny(self, low_threshold, high_threshold, sigma=0.7):
        """
        Canny のエッジ検出を行う
        - scikit-image の canny を使用
        
        Parameters
        ----------
        low_threshold, high_threshold : int
            閾値
        
        sigma : float, default 0.7

        """
        logger = self.logger
        
        self.img = canny(
            (self.img / self.img.max()).astype(np.float32),
            sigma=sigma,
            low_threshold=low_threshold,
            high_threshold=high_threshold
        ).astype(np.uint8)
        
        if logger:
            logger.logging_img(self.img * 255, "canny")
            logger.logging_dict({
                "sigma"         : sigma,
                "low_threshold" : low_threshold,
                "high_threshold": high_threshold,
            }, "params_canny")
        
        self.BG, self.FG = 0, 1
    
    def get_as_image(self):
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
            "branch"  : [0, 0, 255],
            "endpoint": [0, 255, 255]
        }
        
        img = np.zeros(
            (*classified.shape[:2], 3),
            dtype=np.uint8
        )
        
        for type, color in COLORS.items():
            img[classified == LABELS[type]] = color
        
        return img
    
    
    def _classify_pixel(self, _img):
        _FG, _BG = self.FG, self.BG
        _k_size = self.K_SIZE
        _LABELS = self.LABELS
        _TEMPLATE = self.TEMPLATE
        
        _height, _width = _img.shape
        _i = _height // 2
        _pad_width = _k_size // 2 * 2
        
        _classified = np.full(
            _width - _pad_width,
            fill_value=_LABELS["UNDEFINED"]
        )
        
        _worker_id = int(re.match(r"(.*)-([0-9]+)$", current_process().name).group(2))
        _desc = f"Worker #{_worker_id:3d}"
        
        for _jj, _j in tqdm(
            enumerate(
                range(_k_size // 2, _width - (_k_size // 2))
            ),
            position=_worker_id,
            desc=_desc,
            total=_classified.size,
            leave=False
        ):
            
            # 背景画素だったらスキップ
            if _img[_i, _j] == _BG:
                _classified[_jj] = _LABELS["BG"]
                continue
            
            # img_roi: (i, j) を中心に k_size の大きさの矩形
            _img_roi = _img[
                get_window_rect(
                    img_shape=_img.shape,
                    center=(_j, _i),
                    wnd_size=_k_size,
                    ret_type="slice"
                )
            ].copy()
            _img_roi[_img_roi == _FG] = 1
            
            # 「端点」「分岐点」へのマッチング
            for _type in _TEMPLATE.keys():
                for _template in _TEMPLATE[_type]:
                    
                    # もしテンプレートにマッチングしたら
                    if self.match_to_template(_img_roi, _template):
                        
                        # _classified に結果を保持
                        _classified[_jj] = _LABELS[_type]
        
        # テンプレートにマッチしなかったエッジ画素 → 「通過点」
        _classified[_classified == _LABELS["UNDEFINED"]] = _LABELS["passing"]
        
        return _classified
    
    def classify(self):
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
        k_size = self.K_SIZE
        logger = self.logger
        img = self.img
        
        # 幅 1 のパディングを追加
        img = np.pad(
            img, pad_width=k_size // 2,
            mode="constant",
            constant_values=BG
        )
        height, width = img.shape[:2]
        
        progress_bar = tqdm(total=height)
        
        def _update_progressbar(arg):
            progress_bar.update()
        
        pool = Pool(processes=12, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
        
        results = list()
        
        # 全画素ループ (tqdm で進捗可視化)
        for i in trange((k_size // 2), height - (k_size // 2), desc="Height", leave=False):
            
            roi = img[i:i + k_size, :]
            
            results.append(
                pool.apply_async(
                    self._classify_pixel,
                    args=(roi,),
                    callback=_update_progressbar
                )
            )
        pool.close()
        pool.join()
        
        self.classified = np.array(
            [result.get() for result in results],
            dtype=self.classified.dtype
        )
        
        if logger:
            logger.logging_img(self.classified, "classified")
            logger.logging_img(self.get_as_image(), "classified_visualized")
        
        return self.classified
    
    def calc_metrics(self):
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
            (extracted * 255).astype(np.uint8),
            connectivity=8
        )
        
        metrics["average_length"] = np.count_nonzero(extracted) / n_labels
        
        if logger:
            logger.logging_dict(metrics, "metrics")
            logger.logging_img(labels, "edge_lines", cmap="jet")
        
        return metrics
