# /usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2019-11-28
# This is a part of EQDmgAnalyzr


from itertools import product
from multiprocessing import current_process, Pool
import re

import cv2
import numpy as np
from tqdm import tqdm

from utils.assertion import NDARRAY_ASSERT, SAME_SHAPE_ASSERT, TYPE_ASSERT
from utils.evaluation import evaluation_by_confusion_matrix
from utils.common import worker_exception_raisable
from utils.logger import ImageLogger
from utils.pool import CustomPool


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
    
    
    def find_color_threshold_in_hsv(self, img, ground_truth, precision=10):
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
            _worker_id = current_process()._identity[0]
            _desc = f"Worker #{_worker_id:3d}"
            
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
                desc=_desc,
                position=_worker_id,
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
        progress_bar = tqdm(total=len(Q), position=0)
        
        
        def _update_progressbar(arg):
            progress_bar.update()
        
        
        # Initialize process pool
        cp = CustomPool()
        pool = cp.Pool(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
        
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
        cp.update()
        
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
            "Score"           : {
                "F Score": -1,
            },
            "Params"          : {
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
                                "Score"           : _metrics,
                                "Params"          : {
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
