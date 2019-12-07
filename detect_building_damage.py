#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2019-11-28
# This is a part of EQDmgAnalyzr

from os import path
from itertools import product

from skimage.morphology import disk
from tqdm import tqdm

import cv2
import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

from imgproc.edge import EdgeProcedures
from imgproc.utils import compute_by_window
from utils.assertion import NDARRAY_ASSERT, SAME_SHAPE_ASSERT, TYPE_ASSERT
from utils.logger import ImageLogger

from utils.common import eprint
from utils.evaluation import evaluation_by_confusion_matrix

plt.switch_backend("macosx")


class ParamsFinder:
    PRECISION = 10
    
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
        (lower, upper)
            パーセンタイルの範囲
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
    
    def find_color_threshold_in_hsv(self, smoothed_img, ground_truth):
        """
        閾値探索

        Parameters
        ----------
        smoothed_img : numpy.ndarray
            平滑化済み入力画像
            - 8-Bit RGB カラー
        ground_truth : numpy.ndarray
            正解画像
            - 1-Bit (np.bool) 2値化画像
            - 黒：背景、白：被害領域

        Returns
        -------
        dict, numpy.ndarray
            導き出された閾値と閾値処理を行った画像のタプル
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
            list(product(np.linspace(50 / self.PRECISION, 50, self.PRECISION), repeat=6))):
            
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
    
    def find_subtracted_thresholds(self, img_a, img_b, ground_truth):
        """ A & not(B) を計算する """
        NDARRAY_ASSERT(img_a, ndim=2, dtype=np.uint8)
        NDARRAY_ASSERT(img_b, ndim=2, dtype=np.uint8)
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
            list(product(np.linspace(50 / self.PRECISION, 50, self.PRECISION), repeat=4))):
            
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
    
    def find_reasonable_morphology(self, result, ground_truth):
        """
        最適なモルフォロジー処理を模索

        Parameters
        ----------
        result : numpy.ndarray
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
        
        NDARRAY_ASSERT(result, ndim=2, dtype=np.bool)
        NDARRAY_ASSERT(ground_truth, ndim=2, dtype=np.bool)
        TYPE_ASSERT(logger, [None, ImageLogger])
        SAME_SHAPE_ASSERT(result, ground_truth, ignore_ndim=True)
        
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
        
        result = (result * 255).astype(np.uint8)
        
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
                            result,
                            cv2.__dict__[f"MORPH_{operation}"],
                            kernel=kernel,
                            iterations=n_iterations
                        ).astype(bool)
                        
                        cm, metrics = evaluation_by_confusion_matrix(_result, ground_truth)
                        
                        eprint("Score:", metrics["F Score"])
                        
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


class BuildingDamageExtractor:
    
    
    def meanshift_and_color_thresholding(self):
        img = self.img
        ground_truth = self.ground_truth
        NDARRAY_ASSERT(img, ndim=3, dtype=np.uint8)
        NDARRAY_ASSERT(ground_truth, ndim=2, dtype=np.bool)
        SAME_SHAPE_ASSERT(img, ground_truth, ignore_ndim=True)
        
        filter_params = dict(sp=40, sr=50)
        
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
            result=thresholded,
            ground_truth=ground_truth,
        )
        
        logger.logging_img(morphologied, "building_damage")
        img = self.img
    
    
    def edge_angle_variance_and_hpf(self):
        img = self.img
        ground_truth = self.ground_truth
        logger = self.logger
        
        params_window_proc = {
            "Edge Angle Variance": None,
            "High Pass Filter"   : None
        }
        
        def hpf(img, freq):
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
            
            fft = np.fft.fftshift(
                np.fft.fft2(img)
            )
            
            mask = _disk_mask(freq, *img.shape[:2])
            
            fft_masked = fft.copy()
            fft_masked[mask] = 0 + 0j
            
            ifft = np.fft.ifft2(fft_masked)
            
            params_window_proc["High Pass Filter"] = {
                "window_size": 8,
                "step"       : 2,
            }
            
            fd_img = compute_by_window(
                np.abs(ifft),
                lambda img: np.mean(img),
                dst_dtype=np.float64,
                **params_window_proc["High Pass Filter"]
            )
            
            fd_img = ndi.zoom(
                fd_img / fd_img.max(),
                (img.shape[0] / fd_img.shape[0], img.shape[1] / fd_img.shape[1]),
                order=0,
                mode='nearest'
            )
            
            logger.logging_img(np.log10(np.abs(fft)), "power_spector", cmap="jet")
            logger.logging_img(mask, cmap="gray_r")
            logger.logging_img(np.log10(np.abs(fft_masked)), "power_spector_masked", cmap="jet")
            logger.logging_img(np.abs(ifft), "IFFT")
            logger.logging_img(fd_img, "HPF_gray")
            logger.logging_img(fd_img, "HPF_colorized", cmap="jet")
            
            return fd_img
        
        
        params_finder = ParamsFinder(logger=self.logger)
        
        if img.ndim != 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find Canny Thresholds
        _, canny_edge = params_finder.find_canny_thresholds(img, ground_truth)
        
        edge_proc = EdgeProcedures(img)
        edge_proc.edge_magnitude = canny_edge
        
        params_window_proc["Edge Angle Variance"] = {
            "window_size": 8,
            "step"       : 2
        }
        
        # Edge Angle Variance
        fd_variance = edge_proc.get_feature_by_window(
            edge_proc.angle_variance_using_mean_vector,
            **params_window_proc["Edge Angle Variance"]
        )
        
        fd_variance = ndi.zoom(
            fd_variance / fd_variance.max(),
            (img.shape[0] / fd_variance.shape[0], img.shape[1] / fd_variance.shape[1]),
            order=0,
            mode='nearest'
        )
        
        logger.logging_img(edge_proc.edge_magnitude, "magnitude")
        logger.logging_img(edge_proc.edge_angle, "angle", cmap="hsv")
        logger.logging_img(edge_proc.get_angle_colorized_img(), "angle_colorized")
        logger.logging_img(fd_variance, "angle_variance")
        logger.logging_img(fd_variance, "angle_variance", cmap="jet")
        
        # High-Pass Filter
        freq = int(min(img.shape[:2]) * 0.05)
        fd_hpf = hpf(img, freq)
        
        logger.logging_dict(params_window_proc, "params_window_proc")
        
        # Find Thresholds for Angle-Variance Image
        _, result = params_finder.find_subtracted_thresholds(fd_variance, fd_hpf, ground_truth)
        
        logger.logging_img(result, "final_result")
        
        return result
    
    def __init__(self, img, ground_truth, logger=None) -> None:
        super().__init__()
        
        TYPE_ASSERT(img, np.ndarray)
        TYPE_ASSERT(ground_truth, np.ndarray)
        TYPE_ASSERT(logger, [None, ImageLogger])
        
        self.img = img
        self.ground_truth = ground_truth
        self.logger = logger


if __name__ == '__main__':
    # PATH_SRC_IMG = "img/resource/aerial_roi1_raw_ms_40_50.png"
    PATH_SRC_IMG = "img/resource/aerial_roi1_raw_denoised_clipped.png"
    # PATH_SRC_IMG = "img/resource/aerial_roi2_raw.png"
    
    PATH_GT_IMG = "img/resource/ground_truth/aerial_roi1.png"
    # PATH_GT_IMG = "img/resource/ground_truth/aerial_roi2.png"
    
    src_img = cv2.imread(
        PATH_SRC_IMG,
        cv2.IMREAD_COLOR
    )
    
    ground_truth = cv2.imread(
        PATH_GT_IMG,
        cv2.IMREAD_GRAYSCALE
    ).astype(bool)
    
    logger = ImageLogger(
        "./tmp/detect_building_damage",
        suffix=path.splitext(
            path.basename(
                PATH_SRC_IMG
            )
        )[0]
    )
    
    inst = BuildingDamageExtractor(src_img, ground_truth, logger=logger)
    # inst.meanshift_and_color_thresholding()
    inst.edge_angle_variance_and_hpf()
