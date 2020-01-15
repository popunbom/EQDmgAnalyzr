#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2019-11-28
# This is a part of EQDmgAnalyzr
import json
import platform
import re
import sys
from multiprocessing import current_process, Pool
from os import path
from itertools import product
from time import sleep

from scipy.signal.windows import gaussian
from tqdm import tqdm

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pymeanshift

from imgproc.edge import EdgeProcedures
from imgproc.edge_line_feature import EdgeLineFeatures
from imgproc.utils import compute_by_window, zoom_to_img_size, disk_mask
from utils.assertion import NDARRAY_ASSERT, SAME_SHAPE_ASSERT, TYPE_ASSERT
from utils.exception import UnsupportedOption
from utils.logger import ImageLogger

from utils.common import eprint, worker_exception_raisable
from utils.evaluation import evaluation_by_confusion_matrix
from utils.reflection import get_qualified_class_name


if platform.system() == "Darwin":
    plt.switch_backend("macosx")


class ParamsFinder:
    """
    正解データに基づく閾値探索を行う
    
    Attributes
    ----------
    logger : ImageLogger, default None
        ImageLogger インスタンス
        - 求めた閾値や画像をロギングするために使用する
    
    logger_sub_path : str, default "params_finder"
        ImageLogger により生成されるサブフォルダ名
        - 各ロギング結果は `logger_sub_path` のフォルダ名の
          中に格納される
    """
    
    
    @staticmethod
    def _in_range_percentile(arr, q):
        """
        パーセンタイルの範囲を求める
        Parameters
        ----------
        arr : numpy.ndarray
            対象データ
        q : int or tuple
            百分率 ( 0 <= q <= 100 )

        Returns
        -------
        lower_bound, upper_bound : tuple
            パーセンタイルの範囲 (下限, 上限)
        """
        
        if isinstance(q, tuple):
            lower_bound = np.percentile(arr, q=q[0])
            upper_bound = np.percentile(arr, q=(100 - q[1]))
        
        else:
            lower_bound = np.percentile(arr, q=q)
            upper_bound = np.percentile(arr, q=(100 - q))
        
        return lower_bound, upper_bound
    
    
    def __init__(self, logger=None) -> None:
        """
        コンストラクタ

        Parameters
        ----------
        logger : ImageLogger, default is None
            ImageLogger インスタンス
            - 処理途中の画像をロギングする
        """
        
        TYPE_ASSERT(logger, [None, ImageLogger])
        
        super().__init__()
        
        self.logger = logger
        self.logger_sub_path = "params_finder"
    
    
    def find_color_threshold_in_hsv(self, img, ground_truth, precision=10, n_worker=12):
        """
        HSV 色空間における色閾値探索
        
        - RGB → HSV 変換 を行う
        
        - HSV の各チャンネルで閾値処理を行い統合する
        
        - 正解データを用いて精度評価を行う
        

        Parameters
        ----------
        img : numpy.ndarray
            入力画像 (8-Bit RGB カラー)
        ground_truth : numpy.ndarray
            正解データ (1-Bit)
        precision : int
            閾値計算の精度
        

        Returns
        -------
        reasonable_params : dict
            F値が最も高くなったときの閾値
        result : numpy.ndarray
            その際の閾値処理結果画像 (1-Bit 2値画像)
            
        Notes
        -----
        `ground_truth`:
            - 1-Bit (bool 型) 2値画像
            - 黒：背景、白：被害領域
        `precision`:
            - precision=N のとき、H, S, V の各チャンネルに対し
              2N ずつにパーセンタイル分割を行う
        """
        
        global _worker_find_color_threshold_in_hsv
        
        
        # Worker methods executed parallel
        @worker_exception_raisable
        def _worker_find_color_threshold_in_hsv(_img, _masked, _q_h, _q_s):
            # Value used in tqdm
            _WORKER_ID = int(re.match(r"(.*)-([0-9]+)$", current_process().name).group(2))
            _DESC = f"Worker #{_WORKER_ID:3d}"
            
            # Unpack arguments
            _q_h_low, _q_h_high = _q_h
            _q_s_low, _q_s_high = _q_s
            
            # Split image to each channne
            _img_h, _img_s, _img_v = [_img[:, :, i] for i in range(3)]
            _masked_h, _masked_s, _masked_v = [_masked[:, i] for i in range(3)]
            
            # Initialize variables
            reasonable_params = {
                "Score": {
                    "F Score": -1,
                },
                "Range": -1
            }
            
            # Find thresholds
            for _q_v_low, _q_v_high in tqdm(
                list(product(np.linspace(50 / precision, 50, precision), repeat=2)),
                desc=_DESC,
                position=_WORKER_ID,
                leave=False
            ):
                
                # Generate result
                _h_min, _h_max = self._in_range_percentile(_masked_h, (_q_h_low, _q_h_high))
                _s_min, _s_max = self._in_range_percentile(_masked_s, (_q_s_low, _q_s_high))
                _v_min, _v_max = self._in_range_percentile(_masked_v, (_q_v_low, _q_v_high))
                
                _result = (
                    ((_h_min <= _img_h) & (_img_h <= _h_max)) &
                    ((_s_min <= _img_s) & (_img_s <= _s_max)) &
                    ((_v_min <= _img_v) & (_img_v <= _v_max))
                )
                
                # Calculate score
                _cm, _metrics = evaluation_by_confusion_matrix(_result, ground_truth)
                
                # Update reasonable_params
                if _metrics["F Score"] > reasonable_params["Score"]["F Score"]:
                    reasonable_params = {
                        "Score"           : _metrics,
                        "Confusion Matrix": _cm,
                        "Range"           : {
                            "H": (_h_min, _h_max, _q_h_low, _q_h_high),
                            "S": (_s_min, _s_max, _q_s_low, _q_s_high),
                            "V": (_v_min, _v_max, _q_v_low, _q_v_high),
                        }
                    }
            
            return reasonable_params
        
        
        # Check arguments
        NDARRAY_ASSERT(img, ndim=3, dtype=np.uint8)
        NDARRAY_ASSERT(ground_truth, ndim=2, dtype=np.bool)
        SAME_SHAPE_ASSERT(img, ground_truth, ignore_ndim=True)
        
        # Convert RGB -> HSV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # `masked`: `img` masked by `ground_truth`
        masked = img[ground_truth]
        
        # Percentile Split
        Q = list(product(np.linspace(50 / precision, 50, precision), repeat=4))
        
        # `progress_bar`: whole progress bar
        progress_bar = tqdm(total=len(Q))
        
        
        def _update_progressbar(arg):
            progress_bar.update()
        
        
        # Initialize process pool
        pool = Pool(processes=n_worker, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
        
        results = list()
        
        # Multi-Processing !
        for q_h_low, q_h_high, q_s_low, q_s_high in Q:
            results.append(
                pool.apply_async(
                    _worker_find_color_threshold_in_hsv,
                    args=(img, masked, (q_h_low, q_h_high), (q_s_low, q_s_high)),
                    callback=_update_progressbar
                )
            )
        pool.close()
        pool.join()
        
        # Resolve results
        try:
            results = [result.get() for result in results]
        except Exception as e:
            print(e)
        
        # Get result whose F-Score is max in results
        reasonable_params = max(results, key=lambda e: e["Score"]["F Score"])
        
        img_h, img_s, img_v = [img[:, :, i] for i in range(3)]
        h_min, h_max, _, _ = reasonable_params["Range"]["H"]
        s_min, s_max, _, _ = reasonable_params["Range"]["S"]
        v_min, v_max, _, _ = reasonable_params["Range"]["V"]
        
        # Generate image using reasonable thresholds
        result = (
            ((h_min <= img_h) & (img_h <= h_max)) &
            ((s_min <= img_s) & (img_s <= s_max)) &
            ((v_min <= img_v) & (img_v <= v_max))
        )
        
        # Logging
        if self.logger:
            self.logger.logging_dict(reasonable_params, "color_thresholds_in_hsv", sub_path=self.logger_sub_path)
            self.logger.logging_img(result, "meanshift_thresholded", sub_path=self.logger_sub_path)
        
        return reasonable_params, result
    
    
    def find_subtracted_thresholds(self, img_a, img_b, ground_truth, precision=10):
        """
        2画像間の差分結果の閾値計算を行う
        
        - 画像A, B それぞれで閾値処理
        - 各画像の最小値、最大値の間をパーセンタイルで分割する
        - A & not(B) を計算し、正解データと比較
        - F値が最も高くなるときの画像A, B の閾値を返却する
        
        
        Parameters
        ----------
        img_a, img_b :  numpy.ndarray
            入力画像 (グレースケール画像)
        ground_truth : numpy.ndarray
            正解データ (1-Bit)
        precision : int
            閾値計算の精度
        
        Returns
        -------
        reasonable_params : dict
            F値が最も高くなったときの閾値
        result : numpy.ndarray
            その際の閾値処理結果画像 (1-Bit 2値画像)
            
        Notes
        -----
        `ground_truth`:
            - 1-Bit (bool 型) 2値画像
            - 黒：背景、白：被害領域
        `precision`:
            - precision=N のとき、2N ずつにパーセンタイル分割を行う
        
        """
        NDARRAY_ASSERT(img_a, ndim=2)
        NDARRAY_ASSERT(img_b, ndim=2)
        NDARRAY_ASSERT(ground_truth, ndim=2, dtype=np.bool)
        SAME_SHAPE_ASSERT(img_a, img_b, ignore_ndim=False)
        SAME_SHAPE_ASSERT(img_a, ground_truth, ignore_ndim=False)
        
        # Initialize variables
        reasonable_params = {
            "Score"           : {
                "F Score": -1
            },
            "Confusion Matrix": None,
            "Range"           : None,
        }
        result = None
        
        # Calculate thresholds
        for q_a_low, q_a_high, q_b_low, q_b_high in tqdm(
            list(product(np.linspace(50 / precision, 50, precision), repeat=4))):
            
            # Generate result
            a_min, a_max = self._in_range_percentile(img_a, (q_a_low, q_a_high))
            b_min, b_max = self._in_range_percentile(img_b, (q_b_low, q_b_high))
            
            _result_a = (a_min < img_a) & (img_a < a_max)
            _result_b = (b_min < img_b) & (img_b < b_max)
            
            _result = _result_a & np.bitwise_not(_result_b)
            
            # Calculate scores
            _cm, _metrics = evaluation_by_confusion_matrix(_result, ground_truth)
            
            # Update reasonable_params
            if _metrics["F Score"] > reasonable_params["Score"]["F Score"]:
                reasonable_params = {
                    "Score"           : _metrics,
                    "Confusion Matrix": _cm,
                    "Range"           : {
                        "img_a": [a_min, a_max],
                        "img_b": [b_min, b_max],
                    }
                }
                result = _result.copy()
        
        # Logging
        if self.logger:
            self.logger.logging_dict(reasonable_params, f"params_subtracted_thresholds",
                                     sub_path=self.logger_sub_path)
            self.logger.logging_img(result, f"subtract_thresholded", sub_path=self.logger_sub_path)
        
        return reasonable_params, result
    
    
    def find_canny_thresholds(self, img, ground_truth):
        """
        Canny のアルゴリズムの閾値探索を行う
        
        Parameters
        ----------
        img : numpy.ndarray
            入力画像 (8−Bit グレースケール画像)
        ground_truth
            正解データ (1-Bit 画像)

        Returns
        -------
        reasonable_params : dict
            F値が最も高くなったときの閾値
        result : numpy.ndarray
            その際の閾値処理結果画像
            
        Notes
        -----
        `ground_truth`:
            - 1-Bit (bool 型) 2値画像
            - 黒：背景、白：被害領域
        """
        from skimage.feature import canny
        
        NDARRAY_ASSERT(img, ndim=2, dtype=np.uint8)
        NDARRAY_ASSERT(ground_truth, ndim=2, dtype=np.bool)
        SAME_SHAPE_ASSERT(img, ground_truth, ignore_ndim=False)
        
        # Initialize variables
        reasonable_params = {
            "Score"           : {
                "F Score": -1
            },
            "Confusion Matrix": None,
            "Thresholds"      : None,
        }
        result = None
        
        # Calculate thresholds
        for th_1, th_2 in tqdm(list(product(range(256), repeat=2))):
            
            # Generate result
            _result = canny(img, low_threshold=th_1, high_threshold=th_2)
            
            # Calculate scores
            _cm, _metrics = evaluation_by_confusion_matrix(_result, ground_truth)
            
            # Update reasonable_params
            if _metrics["F Score"] > reasonable_params["Score"]["F Score"]:
                reasonable_params = {
                    "Score"           : _metrics,
                    "Confusion Matrix": _cm,
                    "Thresholds"      : list([th_1, th_2])
                }
                result = _result.copy()
        
        if result.dtype != bool:
            result = (result * 255).astype(np.uint8)
        
        # Logging
        if self.logger:
            self.logger.logging_dict(reasonable_params, "canny_thresholds", sub_path=self.logger_sub_path)
            self.logger.logging_img(result, "canny", sub_path=self.logger_sub_path)
        
        return reasonable_params, result
    
    
    def find_reasonable_morphology(self, result_img, ground_truth):
        """
        最適なモルフォロジー処理を模索
        
        - 正解データとの精度比較により、各種処理結果の補正として
          最適なモルフォロジー処理を模索する
          

        Parameters
        ----------
        result_img : numpy.ndarray
            処理結果画像
        ground_truth : numpy.ndarray
            正解データ
        
        Returns
        -------
        reasonable_params : dict
            導き出されたパラメータ
            - モルフォロジー処理のパターン
            - 適用時のスコア
        result : numpy.ndarray
            - モルフォロジー処理結果画像
        
        Notes
        -----
        `result_img`:
            - 1-Bit (bool 型) 2値画像
            - 黒：無被害、白：被害
        
        `ground_truth`:
            - 1-Bit (bool 型) 2値画像
            - 黒：背景、白：被害領域
        """
        
        # Check arguments
        NDARRAY_ASSERT(result_img, ndim=2, dtype=np.bool)
        NDARRAY_ASSERT(ground_truth, ndim=2, dtype=np.bool)
        SAME_SHAPE_ASSERT(result_img, ground_truth, ignore_ndim=True)
        
        # 処理の組み合わせパターン
        FIND_RANGE = {
            # カーネルの大きさ: 3x3, 5x5
            "Kernel Size"       : [3, 5],
            # 処理の対象: 4近傍, 8近傍
            "# of Neighbors"    : [4, 8],
            # モルフォロジー処理
            "Morphology Methods": ["ERODE", "DILATE", "OPEN", "CLOSE"],
            # 繰り返し回数
            "# of Iterations"   : range(1, 6)
        }
        
        # Convert input image to uint8 (for `cv2.morphologyEx`)
        result_img = (result_img * 255).astype(np.uint8)
        
        # Initialize variables
        reasonable_params = {
            "Confusion Matrix": dict(),
            "Score" : {
                "F Score"         : -1,
            },
            "Params": {
                "Operation" : "",
                "Kernel"    : {
                    "Size"         : (-1, -1),
                    "#_of_Neighbor": -1
                },
                "Iterations": -1
            }
        }
        result = None
        
        # Finding reasonable process
        for kernel_size in FIND_RANGE["Kernel Size"]:
            for n_neighbor in FIND_RANGE["# of Neighbors"]:
                for operation in FIND_RANGE["Morphology Methods"]:
                    for n_iterations in FIND_RANGE["# of Iterations"]:
                        
                        # Set parameters
                        if n_neighbor == 4:
                            kernel = np.zeros(
                                (kernel_size, kernel_size),
                                dtype=np.uint8
                            )
                            kernel[kernel_size // 2, :] = 1
                            kernel[:, kernel_size // 2] = 1
                        else:
                            kernel = np.ones(
                                (kernel_size, kernel_size),
                                dtype=np.uint8
                            )
                        
                        # Generate result
                        _result = cv2.morphologyEx(
                            src=result_img,
                            op=cv2.__dict__[f"MORPH_{operation}"],
                            kernel=kernel,
                            iterations=n_iterations
                        ).astype(bool)
                        
                        # Calculate scores
                        _cm, _metrics = evaluation_by_confusion_matrix(_result, ground_truth)
                        
                        # Update reasonable_params
                        if _metrics["F Score"] > reasonable_params["Score"]["F Score"]:
                            
                            reasonable_params = {
                                "Confusion Matrix": _cm,
                                "Score"         : _metrics,
                                "Params": {
                                    "Operation" : operation,
                                    "Kernel"    : {
                                        "Size"         : (kernel_size, kernel_size),
                                        "#_of_Neighbor": n_neighbor
                                    },
                                    "Iterations": n_iterations
                                }
                            }
                            
                            result = _result.copy()
        
        # Logging
        if self.logger:
            self.logger.logging_img(result, "result_morphology", sub_path=self.logger_sub_path)
            self.logger.logging_dict(reasonable_params, "params_morphology", sub_path=self.logger_sub_path)
        
        return reasonable_params, result
    
    
    def find_threshold(self, img, ground_truth, precision=100, logger_suffix=""):
        """
        正解データを用いて最適な閾値を求める
        
        - 各画像の最小値、最大値の間をパーセンタイルで分割し、閾値を生成
        - 閾値処理を行った結果を正解データと比較する
        - F値が最も高くなるときの閾値を返却する
        
        Parameters
        ----------
        img :  numpy.ndarray
            入力画像 (グレースケール画像)
        ground_truth : numpy.ndarray
            正解データ (1-Bit)
        precision : int, default 100
            閾値計算の精度
        logger_suffix : str, default ""
            ImageLogger によるロギングの際に画像が格納される
            フォルダの末尾につく文字列を指定する
        
        Returns
        -------
        reasonable_params : dict
            F値が最も高くなったときの閾値
        result : numpy.ndarray
            その際の閾値処理結果画像 (1-Bit 2値画像)
            
        Notes
        -----
        `ground_truth`:
            - 1-Bit (bool 型) 2値画像
            - 黒：背景、白：被害領域
        `precision`:
            - precision=N のとき、2N ずつにパーセンタイル分割を行う
        
        """
        
        NDARRAY_ASSERT(img, ndim=2)
        NDARRAY_ASSERT(ground_truth, ndim=2, dtype=np.bool)
        TYPE_ASSERT(logger_suffix, str, allow_empty=True)
        
        reasonable_params = {
            "Score"           : {
                "F Score": -1
            },
            "Confusion Matrix": None,
            "Range"           : None,
        }
        result = None
        
        for q_low, q_high in tqdm(list(product(np.linspace(50 / precision, 50, precision), repeat=2))):
            
            v_min, v_max = self._in_range_percentile(img, (q_low, q_high))
            
            _result = (v_min < img) & (img < v_max)
            
            _cm, _metrics = evaluation_by_confusion_matrix(_result, ground_truth)
            
            if _metrics["F Score"] > reasonable_params["Score"]["F Score"]:
                reasonable_params = {
                    "Score"           : _metrics,
                    "Confusion Matrix": _cm,
                    "Range"           : [v_min, v_max]
                }
                result = _result.copy()
        
        if result.dtype != bool:
            result = (result * 255).astype(np.uint8)
        
        if self.logger:
            self.logger.logging_dict(reasonable_params, f"params",
                                     sub_path="_".join([x for x in [self.logger_sub_path, logger_suffix] if x]))
            self.logger.logging_img(result, f"result_thresholded",
                                    sub_path="_".join([x for x in [self.logger_sub_path, logger_suffix] if x]))
        
        return reasonable_params, result


class BuildingDamageExtractor:
    """
    建物被害抽出
    
    Attributes
    ----------
    img : numpy.ndarray
        入力画像
        
    ground_truth : numpy.ndarray
        正解データ
        - 最適な閾値を求めるために使用する
        
    logger : ImageLogger, default None
        ImageLogger インスタンス
        - 求めた閾値や画像をロギングするために使用する
    
    """
    
    
    @staticmethod
    def _f_mean(roi):
        """
        特徴量計算: 行列要素の平均値
        
        Parameters
        ----------
        roi : numpy.ndarray
            局所領域画像
        
        Returns
        -------
        np.float
           特徴量値
        """
        
        return np.mean(roi)
    
    
    @staticmethod
    def _f_calc_percentage(roi):
        """
        特徴量計算: 「端点」「分岐点」の割合
        
        Parameters
        ----------
        roi : numpy.ndarray
            局所領域画像
        
        Returns
        -------
        value : np.float
           特徴量値
        """
        
        LABELS = EdgeLineFeatures.LABELS
        BG = LABELS["BG"]
        ENDPOINT = LABELS["endpoint"]
        BRANCH = LABELS["branch"]
        
        # TODO: 分母の値は「領域サイズ」？それとも「エッジ画素数」？
        n_edges = roi.size - roi[roi == BG].size
        # n_edges = roi.size
        
        if n_edges == 0:
            return 0
        
        n_endpoint = roi[roi == ENDPOINT].size
        n_branch = roi[roi == BRANCH].size
        
        value = (n_endpoint + n_branch) / n_edges
        
        return value
    
    
    @staticmethod
    def _f_calc_weighted_percentage(roi):
        """
        特徴量計算: 「端点」「分岐点」の重み付け割合
        
        - ガウシアンカーネルによる重み付けを行う
        
        Parameters
        ----------
        roi : numpy.ndarray
            局所領域画像
        
        Returns
        -------
        value : np.float
           特徴量値
        """
        LABELS = EdgeLineFeatures.LABELS
        BG = LABELS["BG"]
        ENDPOINT = LABELS["endpoint"]
        BRANCH = LABELS["branch"]
        
        sigma = roi.shape[0] / 3
        
        # Generate Gaussian kernel
        gaussian_kernel = np.outer(
            gaussian(roi.shape[0], std=sigma),
            gaussian(roi.shape[1], std=sigma)
        )
        
        # TODO: 分母の値は「領域サイズ」？それとも「エッジ画素数」？
        # w_n_edges = np.sum(gaussian_kernel) - np.sum(gaussian_kernel[roi == BG])
        w_n_edges = np.sum(gaussian_kernel)
        
        if w_n_edges == 0:
            return 0
        
        w_n_endpoint = np.sum(gaussian_kernel[roi == ENDPOINT])
        w_n_branch = np.sum(gaussian_kernel[roi == BRANCH])
        
        value = (w_n_endpoint + w_n_branch) / w_n_edges
        
        return value
    
    
    @staticmethod
    def calc_edge_angle_variance(img, window_size=33, step=1, logger=None):
        """
        エッジ角度分散を計算
        
        - 入力画像に対し、エッジ抽出を行う
        - ウィンドウ処理を行い、局所領域内におけるエッジ角度の分散を求める
        
        Parameters
        ----------
        img : numpy.ndarray
            入力画像 (グレースケール画像)
        window_size : int, default 33
            ウィンドウ処理におけるウィンドウサイズ
        step : int, default 1
            ウィンドウ処理におけるずらし幅
        logger : ImageLogger, default None
            ImageLogger インスタンス
            
        Returns
        -------
        features : numpy.ndarray
            特徴量画像 (32-Bit float 画像)

        """
        
        # Check arguments
        NDARRAY_ASSERT(img, ndim=2)
        TYPE_ASSERT(window_size, int)
        TYPE_ASSERT(step, int)
        TYPE_ASSERT(logger, [None, ImageLogger])
        
        # Initialize variables
        edge_proc = EdgeProcedures(img)
        
        # Set parameters
        params = {
            "window_proc": {
                "window_size": window_size,
                "step"       : step
            }
        }
        
        # Calculate Edge Angle Variance in window
        features = edge_proc.get_feature_by_window(
            edge_proc.angle_variance_using_mean_vector,
            **params["window_proc"]
        )
        
        # Scale feature image to the same size of input image
        features = zoom_to_img_size(
            # FIXED: Normalize
            features,
            img.shape
        )
        
        # Logging
        if isinstance(logger, ImageLogger):
            logger.logging_dict(params, "params", sub_path="edge_angle_variance")
            logger.logging_img(edge_proc.edge_magnitude, "magnitude", sub_path="edge_angle_variance")
            logger.logging_img(edge_proc.edge_angle, "angle", cmap="hsv", sub_path="edge_angle_variance")
            logger.logging_img(edge_proc.get_angle_colorized_img(), "angle_colorized", sub_path="edge_angle_variance")
            logger.logging_img(features, "angle_variance", sub_path="edge_angle_variance")
            logger.logging_img(features, "angle_variance", cmap="jet", sub_path="edge_angle_variance")
        
        return features
    
    
    @staticmethod
    def high_pass_filter(img, freq=None, window_size=33, step=1, logger=None):
        """
        ハイパスフィルタを適用する
        
        - 画像に対し、離散フーリエ変換を行う
        - 周波数領域上で、ハイパスフィルタを適用する
        - フーリエ逆変換により、高周波領域を抽出する
        - ウィンドウ処理を行い、局所領域内における画素値の平均値を求める
        
        Parameters
        ----------
        img : numpy.ndarray
            入力画像 (8-Bit グレースケール画像)
        freq : float, default None
            ハイパスフィルタの周波数
        window_size : int, default 33
            ウィンドウ処理におけるウィンドウサイズ
        step : int, default 1
            ウィンドウ処理におけるずらし幅
        logger : ImageLogger, default None
            ImageLogger インスタンス
        
        Returns
        -------
        features : numpy.ndarray
            特徴量画像 (32-Bit float 画像)
            
        See Also
        --------
        ハイパスフィルタ
            - 円盤状のマスク画像を適用することで実現する
        """
        
        NDARRAY_ASSERT(img, ndim=2, dtype=np.uint8)
        TYPE_ASSERT(freq, [None, float])
        TYPE_ASSERT(logger, [None, ImageLogger])
        
        freq = freq or int(min(img.shape[:2]) * 0.05)
        
        # Set parameters
        params = {
            "freq"       : freq,
            "window_proc": {
                "window_size": window_size,
                "step"       : step
            }
        }
        
        # fft: 2-D Fourier Matrix
        fft = np.fft.fftshift(
            np.fft.fft2(img)
        )
        
        # mask: High-pass mask
        mask = disk_mask(freq, *img.shape[:2])
        
        # Apply `mask` to `fft`
        # `mask` の値が '1' の部分を 0値(0+0j) にする
        fft_masked = fft.copy()
        fft_masked[mask] = 0 + 0j
        
        # i_fft: invert FFT
        i_fft = np.fft.ifft2(fft_masked)
        
        # Calculate Mean of pixel values in window
        features = compute_by_window(
            np.abs(i_fft),
            BuildingDamageExtractor._f_mean,
            dst_dtype=np.float64,
            **params["window_proc"]
        )
        
        # Scale feature image to the same size of input image
        features = zoom_to_img_size(
            # FIXED: Normalize
            features,
            img.shape
        )
        
        # Logging
        if isinstance(logger, ImageLogger):
            logger.logging_dict(params, "params", sub_path="high_pass_filter")
            logger.logging_img(np.log10(np.abs(fft)), "power_spectrum", cmap="jet", sub_path="high_pass_filter")
            logger.logging_img(mask, "mask", cmap="gray_r", sub_path="high_pass_filter")
            logger.logging_img(np.log10(np.abs(fft_masked)), "power_spectrum_masked", cmap="jet",
                               sub_path="high_pass_filter")
            logger.logging_img(np.abs(i_fft), "IFFT", sub_path="high_pass_filter")
            logger.logging_img(features, "HPF_gray", sub_path="high_pass_filter")
            logger.logging_img(features, "HPF_colorized", cmap="jet", sub_path="high_pass_filter")
        
        return features
    
    
    def meanshift_and_color_thresholding(self,
                                         func_mean_shift=cv2.pyrMeanShiftFiltering,
                                         params_mean_shift={ "sp": 40, "sr": 50 },
                                         retval_pos=None
     ):
        """
        建物被害検出: Mean-Shift による減色処理と色閾値処理
        
        - 入力画像に Mean-Shift による減色処理を適用する
        - 正解データをもとに、色空間での閾値を探索し、適用する
        - 閾値処理結果に対し、モルフォロジー処理による補正処理を行う
        
        Parameters
        ----------
        func_mean_shift : callable object
            Mean-Shift 処理関数
        params_mean_shift : dict
            Mean-Shift 処理関数に渡されるパラメータ
        retval_pos : int, default None
            Mean-Shift 処理関数の返却値が複数の場合、
            領域分割後の画像が格納されている位置を指定する
        
        Returns
        -------
        building_damage : numpy.ndarray
            被害抽出結果
            
        Notes
        -----
        `func_mean_shift`
            - 第1引数に画像を取れるようになっている必要がある
        `building_damage`
            - 1-Bit (bool 型) 2値画像
            - 黒：背景、白：被害抽出結果
        """
        
        img = self.img
        ground_truth = self.ground_truth
        logger = self.logger
        
        # Check arguments
        NDARRAY_ASSERT(img, ndim=3, dtype=np.uint8)
        NDARRAY_ASSERT(ground_truth, ndim=2, dtype=np.bool)
        SAME_SHAPE_ASSERT(img, ground_truth, ignore_ndim=True)
        
        TYPE_ASSERT(params_mean_shift, dict)
        
        if isinstance(func_mean_shift, str):
            func_mean_shift = eval(func_mean_shift)

        assert callable(func_mean_shift), "argument 'func_mean_shift' must be callable object"
        
        # Set parameters
        params = {
            "Mean-Shift": {
                "func"  : get_qualified_class_name(func_mean_shift, wrap_with_quotes=False),
                "params": params_mean_shift
            }
        }
        
        eprint(
            "Pre-processing ({func_name}, {params}) ... ".format(
                func_name=params["Mean-Shift"]["func"],
                params=", ".join([f"{k}={v}" for k, v in params["Mean-Shift"]["params"].items()])
            ),
            end=""
        )
        
        # Mean-Shift
        if retval_pos is None:
            smoothed = func_mean_shift(
                img,
                **params_mean_shift
            )
        else:
            smoothed = func_mean_shift(
                img,
                **params_mean_shift
            )[retval_pos]
        
        eprint("done !")
        
        # Logging
        if logger:
            logger.logging_img(smoothed, "filtered")
            logger.logging_dict(
                params,
                "detail_mean_shift"
            )
        
        params_finder = ParamsFinder(logger=logger)
        
        # Find: Color thresholds in HSV
        _, result = params_finder.find_color_threshold_in_hsv(
            img=smoothed,
            ground_truth=ground_truth,
        )

        # Find: Morphology processing
        _, building_damage = params_finder.find_reasonable_morphology(
            result_img=result,
            ground_truth=ground_truth,
        )
        
        # Logging
        if logger:
            logger.logging_img(building_damage, "building_damage")
        
        return building_damage
    
    
    def edge_angle_variance_with_hpf(self):
        """
        建物被害検出: エッジ角度分散＋ハイパスフィルタ
        
        - 入力画像からエッジ角度分散と、ハイパスフィルタの特徴量
          画像を生成する
        
        - エッジ角度分散、ハイパスフィルタの両結果に閾値処理を行う
        
        - エッジ角度分散の結果からハイパスフィルタの結果を減算し
          建物抽出結果とする
          
        Returns
        -------
        building_damage : numpy.ndarray
            被害抽出結果
            
        Notes
        -----
        `building_damage`
            - 1-Bit (bool 型) 2値画像
            - 黒：背景、白：被害抽出結果
        """
        
        img = self.img_gs
        ground_truth = self.ground_truth
        logger = self.logger
        
        params_finder = ParamsFinder(logger=logger)
        
        # Edge Angle Variance
        eprint("Calculate: Edge Angle Variance")
        fd_variance = self.calc_edge_angle_variance(img, logger=logger)
        
        # High-Pass Filter
        eprint("Calculate: High-Pass Filter")
        fd_hpf = self.high_pass_filter(img, logger=logger)
        
        # TODO: 各結果画像の閾値処理をどうする？
        # Find Thresholds (only AngleVariance)
        eprint("Calculate: Thresholds (AngleVar)")
        params_finder.find_threshold(fd_variance, ground_truth, logger_suffix="angle_variance")
        
        # Find Thresholds (Combination of AngleVariance and HPF)
        eprint("Calculate: Thresholds (AngleVar - HPF)")
        _, building_damage = params_finder.find_subtracted_thresholds(fd_variance, fd_hpf, ground_truth)
        
        # Logging
        if logger:
            logger.logging_img(building_damage, "building_damage")
        
        return building_damage
    
    
    def edge_pixel_classify(self, sigma=0.1, thresholds=(0.2, 0.5), window_size=33, step=1):
        """
        建物被害検出: エッジ画素分類
        
        - 入力画像に Canny のエッジ抽出を適用する
        - 各エッジ画素を「端点」「分岐点」「通過点」に分類する
        - ウィンドウ処理を行い、局所領域内における「端点」「分岐点」の割合を求める
        - 正解データによる閾値処理を適用し、建物被害抽出結果とする
        
        Parameters
        ----------
        sigma : float, default 0.1
            Canny のアルゴリズムにおける平滑化パラメータ
        thresholds : tuple of float, default (0.2, 0.5)
            Canny のアルゴリズムにおける閾値
        window_size : int, default 33
            ウィンドウ処理におけるウィンドウサイズ
        step : int, default 1
            ウィンドウ処理におけるずらし幅

        Returns
        -------
        building_damage : numpy.ndarray
            被害抽出結果
            
        Notes
        -----
        `building_damage`
            - 1-Bit (bool 型) 2値画像
            - 黒：背景、白：被害抽出結果
        """
        img = self.img_gs
        logger = self.logger
        ground_truth = self.ground_truth
        
        low_threshold, high_threshold = thresholds
        
        # Set parameters
        params = {
            "canny"      : {
                "sigma"         : sigma,
                "low_threshold" : low_threshold,
                "high_threshold": high_threshold
            },
            "window_proc": {
                "window_size": window_size,
                "step"       : step
            }
        }
        
        fd = EdgeLineFeatures(img, logger=logger)
        
        # Apply Canny's Edge Detector
        fd.do_canny(**params["canny"])
        
        # Do Edge Pixel Classify
        classified = fd.classify()
        fd.calc_metrics()
        
        # Calculate Proportion of ENDPOINT and BRANCH in window
        features = compute_by_window(
            imgs=classified,
            # func=self._f_calc_percentage,
            func=self._f_calc_weighted_percentage,
            n_worker=12,
            **params["window_proc"]
        )
        
        # Scale feature image to the same size of input image
        features = zoom_to_img_size(
            features,
            img.shape
        )
        
        # Logging
        if logger:
            logger.logging_img(features, f"features")
            logger.logging_img(features, f"features_colorized", cmap="jet")
            
            logger.logging_dict(params, "params")
        
        params_finder = ParamsFinder(logger=logger)
        
        # Find thresholds
        _, building_damage = params_finder.find_threshold(
            features,
            ground_truth,
            logger_suffix="edge_line_feature"
        )
        
        # Logging
        if logger:
            logger.logging_img(building_damage, "building_damage")
        
        return building_damage
    
    
    def __init__(self, img, ground_truth, logger=None) -> None:
        """
        コンストラクタ
        
        Parameters
        ----------
        img : numpy.ndarray
            入力画像
        ground_truth : numpy.ndarray
            正解データ
        logger : ImageLogger
            ImageLogger インスタンス
            - 求めた閾値や画像をロギングするために使用する
        """
        super().__init__()
        
        TYPE_ASSERT(img, np.ndarray)
        TYPE_ASSERT(ground_truth, np.ndarray)
        TYPE_ASSERT(logger, [None, ImageLogger])
        
        self.img = img
        self.ground_truth = ground_truth
        self.logger = logger
        
        self._img_gs = None
        if img.ndim == 2:
            self._img_gs = img
    
    
    @property
    def img_gs(self):
        """ img_gs : 入力画像 (グレースケール) """
        if self._img_gs is None:
            self._img_gs = cv2.cvtColor(
                self.img,
                cv2.COLOR_BGR2GRAY
            )
        
        return self._img_gs


def test_whole_procedures(path_src_img, path_ground_truth, parameters):
    """ すべての手法をテストする """
    
    C_RED = [0, 0, 255]
    C_ORANGE = [0, 127, 255]
    
    IMG = cv2.imread(
        path_src_img,
        cv2.IMREAD_COLOR
    )
    
    GT = cv2.imread(
        path_ground_truth,
        cv2.IMREAD_COLOR
    )
    
    procedures = [
        "meanshift_and_color_thresholding",
        # "edge_angle_variance_with_hpf",
        # "edge_pixel_classify"
    ]
    
    # for gt_opt in ["GT_BOTH", "GT_RED", "GT_ORANGE"]:
    for gt_opt in ["GT_RED", "GT_ORANGE"]:
        if gt_opt == "GT_BOTH":
            ground_truth = np.all(
                (GT == C_RED) | (GT == C_ORANGE),
                axis=2
            )
        elif gt_opt == "GT_RED":
            ground_truth = np.all(
                GT == C_RED,
                axis=2
            )
        else:
            ground_truth = np.all(
                GT == C_ORANGE,
                axis=2
            )
        
        for proc_name in procedures:
            eprint("Do processing:", proc_name)
            logger = ImageLogger(
                "./tmp/detect_building_damage",
                prefix=path.splitext(
                    path.basename(
                        path_src_img
                    )
                )[0],
                suffix=proc_name + "_no_norm_" + gt_opt
            )
            inst = BuildingDamageExtractor(IMG, ground_truth, logger=logger)
            
            if proc_name in parameters:
                inst.__getattribute__(proc_name)(**parameters[proc_name])
            else:
                inst.__getattribute__(proc_name)()


def do_experiment(experiment_parameters):
    
    def extract_parameters(parameters, case_num, procedure_name):
        parameters_by_case = [
            p
            for p in parameters
            if p["experiment_num"] == case_num
        ]
        
        if parameters_by_case:
            parameters_by_case = parameters_by_case[0]
            if procedure_name in parameters_by_case:
                return parameters_by_case[procedure_name]
            
        return dict()
    
    C_RED = [0, 0, 255]
    C_ORANGE = [0, 127, 255]
    
    p = experiment_parameters
    
    for case_num in p["options"]["experiments"]:
        src_img, gt_img = [
            cv2.imread(
                path.join(
                    p["resource_dirs"][img_type],
                    "aerial_roi{n}.png".format(n=case_num)
                ),
                cv2.IMREAD_COLOR
            )
            for img_type in ["aerial_image", "ground_truth"]
        ]
    
        for gt_opt in p["options"]["ground_truth"]:
            
            ground_truth = None
            
            if gt_opt == "GT_BOTH":
                ground_truth = np.all(
                    (gt_img == C_RED) | (gt_img == C_ORANGE),
                    axis=2
                )
            elif gt_opt == "GT_RED":
                ground_truth = np.all(
                    gt_img == C_RED,
                    axis=2
                )
            elif gt_opt == "GT_ORANGE":
                ground_truth = np.all(
                    gt_img == C_ORANGE,
                    axis=2
                )
                
            for procedure_name in p["options"]["procedures"]:
                eprint("Experiment Procedure:", procedure_name)
                
                logger = ImageLogger(
                    p["resource_dirs"]["logging"],
                    prefix="aerial_roi{n}".format(n=case_num),
                    suffix=procedure_name + "_no_norm_" + gt_opt
                )
                inst = BuildingDamageExtractor(src_img, ground_truth, logger=logger)
                
                inst.__getattribute__(procedure_name)(
                    **extract_parameters(p["parameters"], case_num, procedure_name)
                )
                
                # BEEP 5 TIMES (for notification)
                for _ in range(5):
                    sleep(1.0)
                    print("\a", end="", flush=True)
            
            for _ in range(5):
                sleep(1.0)
                print("\a", end="", flush=True)

if __name__ == '__main__':
    
    argc, argv = len(sys.argv), sys.argv
    
    if argc != 2:
        eprint(
            "usage: {prog} [experiment-json-file]".format(
                prog=argv[0]
            )
        )
        sys.exit(-1)
    
    else:
        try:
            do_experiment(json.load(open(argv[1])))
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            with open("error.log", "wt") as f:
                f.write(repr(e))
            # for DEBUG
            raise e
