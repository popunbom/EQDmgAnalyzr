#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2019-11-28
# This is a part of EQDmgAnalyzr

import platform
from os import path
from itertools import product
from time import sleep

from scipy.signal.windows import gaussian
from skimage.morphology import disk
from tqdm import tqdm

import cv2
import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

from imgproc.edge import EdgeProcedures
from imgproc.edge_line_feature import EdgeLineFeatures
from imgproc.utils import compute_by_window, apply_road_mask
from utils.assertion import NDARRAY_ASSERT, SAME_SHAPE_ASSERT, TYPE_ASSERT
from utils.logger import ImageLogger

from utils.common import eprint
from utils.evaluation import evaluation_by_confusion_matrix


if platform.system() == "Darwin":
    plt.switch_backend("macosx")


class ParamsFinder:
    
    
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
            処理途中の画像をロギングする ImageLogger
        """
        
        TYPE_ASSERT(logger, [None, ImageLogger])
        
        super().__init__()
        
        self.logger = logger
        self.logger_sub_path = "params_finder"
    
    
    def find_color_threshold_in_hsv(self, smoothed_img, ground_truth, precision=10):
        """
        HSV 色空間における色閾値探索

        Parameters
        ----------
        smoothed_img : numpy.ndarray
            平滑化済み入力画像 (8-Bit RGB カラー)
        ground_truth : numpy.ndarray
            正解データ (1-Bit)

        Returns
        -------
        reasonable_params : dict
            F値が最も高くなったときの閾値
        result : numpy.ndarray
            その際の閾値処理結果画像
            
        Notes
        -----
        `ground_truth`:
            - 1-Bit (bool 型) の2値化された画像
            - 黒：背景、白：被害領域
        """
        
        NDARRAY_ASSERT(smoothed_img, ndim=3, dtype=np.uint8)
        NDARRAY_ASSERT(ground_truth, ndim=2, dtype=np.bool)
        SAME_SHAPE_ASSERT(smoothed_img, ground_truth, ignore_ndim=True)
        
        img = smoothed_img
        
        # Convert RGB -> HSV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = [img[:, :, i] for i in range(3)]
        
        reasonable_params = {
            "Score": -1,
            "Range": -1
        }
        
        masked = img[ground_truth]
        result = None
        
        for q_h_low, q_h_high, q_s_low, q_s_high, q_v_low, q_v_high in tqdm(
            list(product(np.linspace(50 / precision, 50, precision), repeat=6))
        ):
            
            h_min, h_max = self._in_range_percentile(masked[:, 0], (q_h_low, q_h_high))
            s_min, s_max = self._in_range_percentile(masked[:, 1], (q_s_low, q_s_high))
            v_min, v_max = self._in_range_percentile(masked[:, 2], (q_v_low, q_v_high))
            
            _result = (
                ((h_min <= h) & (h <= h_max)) &
                ((s_min <= s) & (s <= s_max)) &
                ((v_min <= v) & (v <= v_max))
            )
            
            cm, metrics = evaluation_by_confusion_matrix(_result, ground_truth)
            
            if metrics["F Score"] > reasonable_params["Score"]:
                reasonable_params = {
                    "Score"           : metrics["F Score"],
                    "Confusion Matrix": cm,
                    "Range"           : {
                        "H": (h_min, h_max, q_h_low, q_h_high),
                        "S": (s_min, s_max, q_s_low, q_s_high),
                        "V": (v_min, v_max, q_v_low, q_v_high),
                    }
                }
                result = _result.copy()
        
        if self.logger:
            self.logger.logging_dict(reasonable_params, "color_thresolds_in_hsv", sub_path=self.logger_sub_path)
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
            その際の閾値処理結果画像
            
        Notes
        -----
        `ground_truth`:
            - 1-Bit (bool 型) の2値化された画像
            - 黒：背景、白：被害領域
        `precision`:
            - precision=N のとき、2N ずつにパーセンタイル分割を行う
        
        """
        NDARRAY_ASSERT(img_a, ndim=2)
        NDARRAY_ASSERT(img_b, ndim=2)
        NDARRAY_ASSERT(ground_truth, ndim=2, dtype=np.bool)
        SAME_SHAPE_ASSERT(img_a, img_b, ignore_ndim=False)
        SAME_SHAPE_ASSERT(img_a, ground_truth, ignore_ndim=False)
        
        reasonable_params = {
            "Score"           : -1,
            "Confusion Matrix": None,
            "Range"           : None,
        }
        result = None
        
        for q_a_low, q_a_high, q_b_low, q_b_high in tqdm(
            list(product(np.linspace(50 / precision, 50, precision), repeat=4))):
            
            a_min, a_max = self._in_range_percentile(img_a, (q_a_low, q_a_high))
            b_min, b_max = self._in_range_percentile(img_b, (q_b_low, q_b_high))
            
            _result_a = (a_min < img_a) & (img_a < a_max)
            _result_b = (b_min < img_b) & (img_b < b_max)
            
            _result = _result_a & np.bitwise_not(_result_b)
            
            cm, metrics = evaluation_by_confusion_matrix(_result, ground_truth)
            
            if metrics["F Score"] > reasonable_params["Score"]:
                reasonable_params = {
                    "Score"           : metrics["F Score"],
                    "Confusion Matrix": cm,
                    "Range"           : {
                        "img_a": [a_min, a_max],
                        "img_b": [b_min, b_max],
                    }
                }
                result = _result.copy()
        
        if result.dtype != bool:
            result = (result * 255).astype(np.uint8)
        
        if self.logger:
            self.logger.logging_dict(reasonable_params, f"params_subtracted_thresholds",
                                     sub_path=self.logger_sub_path)
            self.logger.logging_img(result, f"subtact_thresholded", sub_path=self.logger_sub_path)
        
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
        reasonable_params, result
        - reasonable_params:
        
        """
        from skimage.feature import canny
        
        NDARRAY_ASSERT(img, ndim=2, dtype=np.uint8)
        NDARRAY_ASSERT(ground_truth, ndim=2, dtype=np.bool)
        SAME_SHAPE_ASSERT(img, ground_truth, ignore_ndim=False)
        
        reasonable_params = {
            "Score"           : -1,
            "Confusion Matrix": None,
            "Thresholds"      : None,
        }
        result = None
        
        for th_1, th_2 in tqdm(list(product(range(256), repeat=2))):
            _result = canny(img, low_threshold=th_1, high_threshold=th_2)
            
            cm, metrics = evaluation_by_confusion_matrix(_result, ground_truth)
            
            if metrics["F Score"] > reasonable_params["Score"]:
                reasonable_params = {
                    "Score"           : metrics["F Score"],
                    "Confusion Matrix": cm,
                    "Thresholds"      : list([th_1, th_2])
                }
                result = _result.copy()
        
        if result.dtype != bool:
            result = (result * 255).astype(np.uint8)
        
        if self.logger:
            self.logger.logging_dict(reasonable_params, "canny_thresholds", sub_path=self.logger_sub_path)
            self.logger.logging_img(result, "canny", sub_path=self.logger_sub_path)
        
        return reasonable_params, result
    
    def find_reasonable_morphology(self, img, ground_truth):
        """
        最適なモルフォロジー処理を模索

        Parameters
        ----------
        img : numpy.ndarray
            結果画像
            - 1-Bit (np.bool) 2値化画像
            - 黒：背景、白：被害領域
        ground_truth : numpy.ndarray
            正解画像
            - 1-Bit (np.bool) 2値化画像
            - 黒：背景、白：被害領域

        Returns
        -------
        dict, numpy.ndarray
            導き出されたパラメータとモルフォロジー処理結果画像のタプル
        """
        
        NDARRAY_ASSERT(img, ndim=2, dtype=np.bool)
        NDARRAY_ASSERT(ground_truth, ndim=2, dtype=np.bool)
        TYPE_ASSERT(logger, [None, ImageLogger])
        SAME_SHAPE_ASSERT(img, ground_truth, ignore_ndim=True)
        
        FIND_RANGE = {
            "Kernel Size"       : [3, 5],
            "# of Neighbors"    : [4, 8],
            "Morphology Methods": ["ERODE", "DILATE", "OPEN", "CLOSE"],
            "# of Iterations"   : range(1, 6)
        }
        
        reasonable_params = {
            "Score" : {
                "Confusion Matrix": -1,
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
        
        img = (img * 255).astype(np.uint8)
        
        result = None
        for kernel_size in FIND_RANGE["Kernel Size"]:
            for n_neighbor in FIND_RANGE["# of Neighbors"]:
                for operation in FIND_RANGE["Morphology Methods"]:
                    for n_iterations in FIND_RANGE["# of Iterations"]:
                        
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
                        
                        _result = cv2.morphologyEx(
                            src=img,
                            op=cv2.__dict__[f"MORPH_{operation}"],
                            kernel=kernel,
                            iterations=n_iterations
                        ).astype(bool)
                        
                        cm, metrics = evaluation_by_confusion_matrix(_result, ground_truth)
                        
                        if metrics["F Score"] > reasonable_params["Score"]["F Score"]:
                            
                            reasonable_params = {
                                "Score" : {
                                    "Confusion Matrix": cm,
                                    "F Score"         : metrics["F Score"],
                                },
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
        
        if self.logger:
            self.logger.logging_img(result, "result_morphology", sub_path=self.logger_sub_path)
            self.logger.logging_dict(reasonable_params, "params_morphology", sub_path=self.logger_sub_path)
        
        return reasonable_params, result
    
    
    def find_threshold(self, img, ground_truth, logger_suffix="", precision=100):
        NDARRAY_ASSERT(img, ndim=2)
        NDARRAY_ASSERT(ground_truth, ndim=2, dtype=np.bool)
        TYPE_ASSERT(logger_suffix, str)
        
        reasonable_params = {
            "Score"           : -1,
            "Confusion Matrix": None,
            "Range"           : None,
        }
        result = None
        
        for q_low, q_high in tqdm(list(product(np.linspace(50 / precision, 50, precision), repeat=2))):
            
            v_min, v_max = self._in_range_percentile(img, (q_low, q_high))
            
            _result = (v_min < img) & (img < v_max)
            
            cm, metrics = evaluation_by_confusion_matrix(_result, ground_truth)
            
            if metrics["F Score"] > reasonable_params["Score"]:
                reasonable_params = {
                    "Score"           : metrics["F Score"],
                    "Confusion Matrix": cm,
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
    
    
    @staticmethod
    def _mean(roi):
        return np.mean(roi)


    @staticmethod
    def calc_percentage(roi):
        LABELS = EdgeLineFeatures.LABELS
        BG = LABELS["BG"]
        ENDPOINT = LABELS["endpoint"]
        BRANCH = LABELS["branch"]
        
        n_edges = roi.size - roi[roi == BG].size
        # n_edges = roi.size
        
        if n_edges == 0:
            return 0
        
        n_endpoint = roi[roi == ENDPOINT].size
        n_branch = roi[roi == BRANCH].size
        
        return (n_endpoint + n_branch) / n_edges


    @staticmethod
    def calc_weighted_percentage(roi):
        LABELS = EdgeLineFeatures.LABELS
        BG = LABELS["BG"]
        ENDPOINT = LABELS["endpoint"]
        BRANCH = LABELS["branch"]
        
        # sigma = params["calc_weighted_percentage"]["sigma"]
        sigma = roi.shape[0] / 3
        
        gaussian_kernel = np.outer(
            gaussian(roi.shape[0], std=sigma),
            gaussian(roi.shape[1], std=sigma)
        )
        
        # w_edges = np.sum(gaussian_kernel) - np.sum(gaussian_kernel[roi == BG])
        w_edges = np.sum(gaussian_kernel)
        
        if w_edges == 0:
            return 0
        
        w_endpoint = np.sum(gaussian_kernel[roi == ENDPOINT])
        w_branch = np.sum(gaussian_kernel[roi == BRANCH])
        
        return (w_endpoint + w_branch) / w_edges


    @staticmethod
    def calc_edge_angle_variance(img, window_size=33, step=1, logger=None):
        NDARRAY_ASSERT(img, ndim=2)
        TYPE_ASSERT(window_size, int)
        TYPE_ASSERT(step, int)
        TYPE_ASSERT(logger, [None, ImageLogger])
        
        # Find Canny Thresholds
        # params_finder = ParamsFinder(logger=self.logger)
        # _, canny_edge = params_finder.find_canny_thresholds(img, ground_truth)
        
        edge_proc = EdgeProcedures(img)
        # edge_proc.edge_magnitude = canny_edge
        
        params = {
            "window_proc": {
                "window_size": window_size,
                "step"       : step
            }
        }
        
        # Edge Angle Variance
        fd_img = edge_proc.get_feature_by_window(
            edge_proc.angle_variance_using_mean_vector,
            **params["window_proc"]
        )
        
        fd_img = ndi.zoom(
            fd_img / fd_img.max(),
            (img.shape[0] / fd_img.shape[0], img.shape[1] / fd_img.shape[1]),
            order=0,
            mode='nearest'
        )
        
        if isinstance(logger, ImageLogger):
            logger.logging_dict(params, "params", sub_path="edge_angle_variance")
            logger.logging_img(edge_proc.edge_magnitude, "magnitude", sub_path="edge_angle_variance")
            logger.logging_img(edge_proc.edge_angle, "angle", cmap="hsv", sub_path="edge_angle_variance")
            logger.logging_img(edge_proc.get_angle_colorized_img(), "angle_colorized", sub_path="edge_angle_variance")
            logger.logging_img(fd_img, "angle_variance", sub_path="edge_angle_variance")
            logger.logging_img(fd_img, "angle_variance", cmap="jet", sub_path="edge_angle_variance")
        
        return fd_img


    @staticmethod
    def high_pass_filter(img, freq=None, window_size=33, step=1, logger=None):
        
        def _disk_mask(r, h, w):
            mask = disk(r)
            p_h, p_w = (h - mask.shape[0], w - mask.shape[1])
            mask = np.pad(
                mask,
                [(
                    (p_h) // 2,
                    (p_h) // 2 + (p_h % 2)
                ), (
                    (p_w) // 2,
                    (p_w) // 2 + (p_w % 2)
                )],
                'constant'
            ).astype(bool)
            
            return mask


        NDARRAY_ASSERT(img, ndim=2, dtype=np.uint8)
        TYPE_ASSERT(freq, [None, float])
        TYPE_ASSERT(logger, [None, ImageLogger])
        
        freq = freq or int(min(img.shape[:2]) * 0.05)
        
        fft = np.fft.fftshift(
            np.fft.fft2(img)
        )
        mask = _disk_mask(freq, *img.shape[:2])
        
        fft_masked = fft.copy()
        fft_masked[mask] = 0 + 0j
        
        i_fft = np.fft.ifft2(fft_masked)
        
        params = {
            "freq"       : freq,
            "window_proc": {
                "window_size": window_size,
                "step"       : step
            }
        }
        
        fd_img = compute_by_window(
            np.abs(i_fft),
            BuildingDamageExtractor._mean,
            dst_dtype=np.float64,
            **params["window_proc"]
        )
        
        fd_img = ndi.zoom(
            fd_img / fd_img.max(),
            (img.shape[0] / fd_img.shape[0], img.shape[1] / fd_img.shape[1]),
            order=0,
            mode='nearest'
        )
        
        if isinstance(logger, ImageLogger):
            logger.logging_dict(params, "params", sub_path="high_pass_filter")
            logger.logging_img(np.log10(np.abs(fft)), "power_spectrum", cmap="jet", sub_path="high_pass_filter")
            logger.logging_img(mask, "mask", cmap="gray_r", sub_path="high_pass_filter")
            logger.logging_img(np.log10(np.abs(fft_masked)), "power_spectrum_masked", cmap="jet",
                               sub_path="high_pass_filter")
            logger.logging_img(np.abs(i_fft), "IFFT", sub_path="high_pass_filter")
            logger.logging_img(fd_img, "HPF_gray", sub_path="high_pass_filter")
            logger.logging_img(fd_img, "HPF_colorized", cmap="jet", sub_path="high_pass_filter")
        
        return fd_img


    def meanshift_and_color_thresholding(self, sp=40, sr=50):
        img = self.img
        ground_truth = self.ground_truth
        NDARRAY_ASSERT(img, ndim=3, dtype=np.uint8)
        NDARRAY_ASSERT(ground_truth, ndim=2, dtype=np.bool)
        SAME_SHAPE_ASSERT(img, ground_truth, ignore_ndim=True)
        
        filter_params = dict(sp=sp, sr=sr)
        
        eprint(
            "Pre-processing (Mean-Shift, {params}) ... ".format(
                params=", ".join([f"{k}={v}" for k, v in filter_params.items()])
            ),
            end=""
        )
        smoothed = cv2.pyrMeanShiftFiltering(img, **filter_params)
        eprint("done !")
        
        if self.logger:
            self.logger.logging_img(smoothed, "filtered")
            self.logger.logging_dict(
                dict(
                    type="cv2.pyrMeanShiftFiltering",
                    **filter_params
                ),
                "filter_detail"
            )
        
        params_finder = ParamsFinder(logger=self.logger)
        
        _, thresholded = params_finder.find_color_threshold_in_hsv(
            smoothed_img=smoothed,
            ground_truth=ground_truth,
        )
        
        _, morphologied = params_finder.find_reasonable_morphology(
            img=thresholded,
            ground_truth=ground_truth,
        )
        
        logger.logging_img(morphologied, "building_damage")
        img = self.img


    def edge_angle_variance_with_hpf(self):
        img = self.img
        ground_truth = self.ground_truth
        logger = self.logger
        
        if img.ndim != 2:
            img = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY
            )

        params_finder = ParamsFinder(logger=logger)
        
        # Edge Angle Variance
        eprint("Calculate: Edge Angle Variance")
        fd_variance = self.calc_edge_angle_variance(img, logger=logger)

        # Find Thresholds (only AngleVariance)
        eprint("Calculate: Thresholds (AngleVar)")
        params_finder.find_threshold(fd_variance, ground_truth)
        
        # High-Pass Filter
        eprint("Calculate: High-Pass Filter")
        fd_hpf = self.high_pass_filter(img, logger=logger)
        
        # Find Thresholds (Combination of AngleVariance and HPF)
        eprint("Calculate: Thresholds (AngleVar - HPF)")
        _, result = params_finder.find_subtracted_thresholds(fd_variance, fd_hpf, ground_truth)
        
        if logger:
            logger.logging_img(result, "building_damage")
        
        return result


    def edge_pixel_classify(self, window_size=33, step=1):
        img = self.img
        logger = self.logger
        ground_truth = self.ground_truth
        
        if img.ndim != 2:
            img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        
        fd = EdgeLineFeatures(img, logger=logger)
        
        params = {
            "canny"      : {
                "sigma"         : 0.1,
                "low_threshold" : 0.2,
                "high_threshold": 0.5
            },
            "window_proc": {
                "window_size": window_size,
                "step"       : step
            }
        }
        
        fd.do_canny(**params["canny"])
        
        classified = fd.classify()
        fd.calc_metrics()
        
        features = compute_by_window(
            imgs=classified,
            # func=self.calc_percentage,
            func=self.calc_weighted_percentage,
            n_worker=12,
            **params["window_proc"]
        )
        
        if logger:
            logger.logging_img(features, f"features")
            logger.logging_img(features, f"features_colorized", cmap="jet")
            
            logger.logging_dict(params, "params")

        params_finder = ParamsFinder(logger=logger)
        _, result = params_finder.find_threshold(features, ground_truth, logger_suffix="edge_line_feature")
        
        if logger:
            logger.logging_img(result, "building_damage")
        
        return result


    def __init__(self, img, ground_truth, logger=None) -> None:
        super().__init__()
        
        TYPE_ASSERT(img, np.ndarray)
        TYPE_ASSERT(ground_truth, np.ndarray)
        TYPE_ASSERT(logger, [None, ImageLogger])
        
        self.img = img
        self.ground_truth = ground_truth
        self.logger = logger


def test_whole_procedures(path_src_img, path_ground_truth):
    src_img = cv2.imread(
        path_src_img,
        cv2.IMREAD_COLOR
    )
    
    ground_truth = cv2.imread(
        path_ground_truth,
        cv2.IMREAD_GRAYSCALE
    ).astype(bool)
    
    procedures = [
        "meanshift_and_color_thresholding",
        "edge_angle_variance_with_hpf",
        "edge_pixel_classify"
    ]

    for proc_name in procedures:
        eprint("Do processing:", proc_name)
        logger = ImageLogger(
            "./tmp/detect_building_damage",
            prefix=path.splitext(
                path.basename(
                    path_src_img
                )
            )[0],
            suffix=proc_name
        )
        inst = BuildingDamageExtractor(src_img, ground_truth, logger=logger)
        inst.__getattribute__(proc_name)()


if __name__ == '__main__':
    # PATH_SRC_IMG = "img/resource/aerial_roi1_raw_ms_40_50.png"
    # PATH_SRC_IMG = "img/resource/aerial_roi1_raw_denoised_clipped.png"
    PATH_SRC_IMG = "img/resource/aerial_roi2_raw.png"
    
    # PATH_GT_IMG = "img/resource/ground_truth/aerial_roi1.png"
    PATH_GT_IMG = "img/resource/ground_truth/aerial_roi2.png"

    # PATH_ROAD_MASK = "img/resource/road_mask/aerial_roi1.png"
    # PATH_ROAD_MASK = "img/resource/road_mask/aerial_roi2.png"

    # test_whole_procedures(PATH_SRC_IMG, PATH_GT_IMG)
    
    # src_img = cv2.imread(
    #     PATH_SRC_IMG,
    #     cv2.IMREAD_COLOR
    # )
    
    # ground_truth = cv2.imread(
    #     PATH_GT_IMG,
    #     cv2.IMREAD_GRAYSCALE
    # ).astype(bool)

    # road_mask = cv2.imread(
    #     PATH_ROAD_MASK,
    #     cv2.IMREAD_GRAYSCALE
    # ).astype(bool)
    # src_img = apply_road_mask(src_img, road_mask)


    inst = BuildingDamageExtractor(src_img, ground_truth, logger=logger)
    # inst.meanshift_and_color_thresholding()
    # inst.edge_angle_variance_with_hpf()
    # inst.edge_pixel_classify()

    for _ in range(5):
        sleep(1.0)
        print("\a")
