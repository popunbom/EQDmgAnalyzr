# /usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: popunbom <fantom0779@gmail.com>
# Created At: 2020/01/17

import cv2
import numpy as np
from scipy.signal.windows import gaussian
import pymeanshift

from damage_detector.params_finder import ParamsFinder
from imgproc.edge import EdgeProcedures
from imgproc.edge_line_feature import EdgeLineFeatures
from imgproc.utils import compute_by_window, zoom_to_img_size, disk_mask, check_if_binary_image

from utils.assertion import NDARRAY_ASSERT, SAME_SHAPE_ASSERT, TYPE_ASSERT
from utils.common import eprint
from utils.logger import ImageLogger
from utils.reflection import get_qualified_class_name


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
    def _remove_tiny_area(img, threshold_area=50):
        """
        微小領域の削除
        
        Parameters
        ----------
        img : numpy.ndarray
            入力画像 (8-bit 二値化画像)
        threshold_area : int, default 50
            面積の閾値
        
        Returns
        -------
        thresholded : numpy.ndarray
            閾値以下の面積を持つ領域を除去した画像

        """
        
        NDARRAY_ASSERT(img, dtype=np.uint8)
        
        assert check_if_binary_image(img), \
            "argument 'img' must be binary image"
    
        _, labels, stats, _ = cv2.connectedComponentsWithStats(img)
    
        areas = stats[:, cv2.CC_STAT_AREA]
    
        area_image = labels.copy()
        for label_num, area in enumerate(areas):
            area_image[area_image == label_num] = area
    
        thresholded = (area_image != areas[0]) & (area_image > threshold_area)
        
        return thresholded
    
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
