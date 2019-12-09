# /usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: popunbom <fantom0779@gmail.com>
# Created At: 2019/06/21

from os import path
import cv2
import numpy as np
from scipy.stats import entropy

from utils.assertion import NDARRAY_ASSERT, TYPE_ASSERT
from utils.logger import ImageLogger
from utils.common import dec_debug, eprint
from utils.exception import InvalidImageOrFile
from utils.mpl import show_images

from .edge import EdgeProcedures

from utils.common import check_module_avaliable

MAMBA_AVAILABLE = check_module_avaliable( "mamba" )


class RegionSegmentation:
    LINE_PIXEL_VAL = np.uint8( 255 )
    LINE_LABEL_VAL = np.int16( 0 )
    
    @dec_debug
    def _watershed( self ):
        """
        Watershed 法による領域分割
        
        Returns
        -------
        numpy.ndarray
            領域分割結果
        """
        kernel = np.array( [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ], dtype=np.uint8 )

        img = self.src_img
        img_gs = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )

        thresh, img_bin = cv2.threshold( img_gs, 0, 255, cv2.THRESH_OTSU )
        eprint( f"::Watershed::  Threshold: {thresh}" )

        # Invert Foreground and Background
        if np.sum( img_bin == 0 ) > np.sum( img_bin == 255 ):
            img_bin = cv2.bitwise_not( img_bin )

        # Sure Background
        img_morph = img_bin
        img_morph = cv2.morphologyEx(
            img_morph, cv2.MORPH_CLOSE, kernel, iterations=1
        )
        img_morph = cv2.morphologyEx(
            img_morph, cv2.MORPH_OPEN, kernel, iterations=1
        )
        sure_bg = img_morph

        # Distance
        dist_transform = cv2.distanceTransform( img_morph, cv2.DIST_L2, 5 )

        # Sure Foreground
        thresh, sure_fg = cv2.threshold(
            dist_transform, 0.5 * dist_transform.max(), 255, 0 )
        sure_fg = sure_fg.astype( np.uint8 )

        # Unknown Area
        unknown = cv2.subtract( sure_bg, sure_fg )

        show_images(
            [img_gs, img_bin, img_morph, dist_transform, sure_fg, unknown],
            list_title=[
                "Grayscale", "Binary", "Morph", "Distance",
                "Foreground", "Unknown"
            ],
            plt_title="Temporary Images",
            logger=self.logger
        )

        # Marker
        _, markers = cv2.connectedComponents( sure_fg )
        markers += 1
        markers[unknown == 255] = 0

        # Watershed
        watershed = cv2.watershed( img, markers )

        show_images(
            [img, img_bin, watershed],
            list_title=["Input", "Binary", "Watershed"],
            list_cmap=['gray', 'rainbow'],
            tuple_shape=(1, 3),
            plt_title="Result",
            logger=self.logger
        )

        return watershed

    if MAMBA_AVAILABLE:
        @dec_debug
        def _watershed_using_quasi_distance( self ):
            """
            疑似ユークリッド距離(Quasi Distance) に基づく
            Watershed 領域分割
            
            Returns
            -------
            numpy.ndarray
                領域分割線の画像
            """
        
            from mamba import (
                imageMb, gradient, add, negate,
                quasiDistance, copyBytePlane,
                subConst, build, maxima, label,
                watershedSegment, logic, mix,
            )
        
            from utils.convert import mamba2np, np2mamba
        
        
            # Channel Split
            if self.src_img.ndim == 3:
                b, g, r = [np2mamba( self.src_img[:, :, i] ) for i in range( 3 )]
            elif self.src_img.ndim == 2:
                b, g, r = [np2mamba( self.src_img )] * 3
        
            # We will perform a thick gradient on each color channel (contours in original
            # picture are more or less fuzzy) and we add all these gradients
            gradient = imageMb( r )
            tmp_1 = imageMb( r )
            gradient.reset()
            gradient( r, tmp_1, 2 )
            add( tmp_1, gradient, gradient )
            gradient( g, tmp_1, 2 )
            add( tmp_1, gradient, gradient )
            gradient( b, tmp_1, 2 )
            add( tmp_1, gradient, gradient )
        
            # Then we invert the gradient image and we compute its quasi-distance
            quasi_dist = imageMb( gradient, 32 )
            negate( gradient, gradient )
            quasiDistance( gradient, tmp_1, quasi_dist )
        
            if self.is_logging:
                self.logger.logging_img( tmp_1, "quasi_dist_gradient" )
                self.logger.logging_img( quasi_dist, "quasi_dist" )
        
            # The maxima of the quasi-distance are extracted and filtered (too close maxima,
            # less than 6 pixels apart, are merged)
            tmp_2 = imageMb( r )
            marker = imageMb( gradient, 1 )
            copyBytePlane( quasi_dist, 0, tmp_1 )
            subConst( tmp_1, 3, tmp_2 )
            build( tmp_1, tmp_2 )
            maxima( tmp_2, marker )
        
            # The marker-controlled watershed of the gradient is performed
            watershed = imageMb( gradient )
            label( marker, quasi_dist )
            negate( gradient, gradient )
            watershedSegment( gradient, quasi_dist )
            copyBytePlane( quasi_dist, 3, watershed )
        
            # The segmented binary and color image are stored
            logic( r, watershed, r, "sup" )
            logic( g, watershed, g, "sup" )
            logic( b, watershed, b, "sup" )
        
            segmented_image = mix( r, g, b )
        
            if self.is_logging:
                self.logger.logging_img( segmented_image, "segmented_image" )
        
            watershed = mamba2np( watershed )
        
            return watershed
    
    @dec_debug
    def labeling_by_segmented_img( self ):
        """
        
        領域分割線画像をラベリング
        
        Returns
        -------
        numpy.ndarray
            ラベリング結果(dtype=np.int)
        """
        
        img = self.segmented_line_img
        
        n_labels, labels = cv2.connectedComponents(
            # img => 線画素: 255
            np.bitwise_not( img ),
            connectivity=4
        )
        self.n_labels = n_labels
        
        return labels
    
    @dec_debug
    def binalization_labels( self, labels, thresh=0, dtype=np.uint8, max_val=255 ):
        """
        ラベリング結果の2値化処理
        閾値 thresh 以下の値を 0、閾値 thresh より大きい値を max_val にする
        
        Parameters
        ----------
        labels : numpy.ndarray
            ラベリング結果データ
        thresh : int, default 0
            閾値
        dtype : type, default np.uint8
            2値化結果のデータ型
        max_val : int, default 255
            2値化の際の最大値

        Returns
        -------
        numpy.ndarray
            2値化済みのデータ
        """
        NDARRAY_ASSERT( labels, ndim=2 )
        
        bin_img = np.zeros( labels.shape, dtype=dtype )
        
        bin_img[labels > thresh] = max_val
        
        return bin_img
    
    @dec_debug
    def get_segmented_image_with_label( self, font_color=(0, 255, 255) ):
        """
        領域分割線画像をラベリングし、ラベル番号付きの
        画像を生成する
        
        Parameters
        ----------
        font_color : tuple of int
            ラベル番号の色
            (B, G, R) ではなく、(R, G, B) なので注意

        Returns
        -------
        numpy.ndarray
            ラベル番号付きの領域分割線画像
        """
        
        base_img = cv2.resize(
            self.src_img,
            dsize=None,
            fx=3.0,
            fy=3.0,
            interpolation=cv2.INTER_NEAREST
        )
        line_img = cv2.resize(
            self.segmented_line_img,
            dsize=None,
            fx=3.0,
            fy=3.0,
            interpolation=cv2.INTER_NEAREST
        )
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            np.bitwise_not( line_img ), connectivity=4
        )
        
        line_img = cv2.cvtColor( line_img, cv2.COLOR_GRAY2BGR )
        
        img_with_label = np.maximum( base_img, line_img )
        
        for label_num, centroid in enumerate( centroids ):
            cv2.putText(
                img=img_with_label,
                text=str( label_num ),
                org=tuple( centroid.astype( np.int16 ) ),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.5,
                color=font_color,
                thickness=1,
                lineType=cv2.LINE_AA
            )
        
        if self.is_logging:
            self.logger.logging_img( img_with_label, "segmented_with_label" )
        
        return img_with_label
    
    @dec_debug
    def calc_region_connection( self ):
        """
        ラベリング結果から隣接領域情報を生成する

        Returns
        -------
        dict
            隣接領域の情報
        """
        
        import queue
        import itertools
        
        # D: differential of points
        D = [
            (0, 1), (-1, 1), (-1, 0), (-1, -1),
            (0, -1), (1, -1), (1, 0), (1, 1)
        ]
        # D = [
        #     (0, 1), (-1, 0), (0, -1), (1, 0)
        # ]
        
        line_img = self.segmented_line_img
        labels = self.labels
        relation = dict()
        checked = np.zeros( labels.shape, dtype=np.bool )
        
        loop_count = 0
        while True:
            unchecked_line_pixels = np.argwhere(
                (line_img == self.LINE_PIXEL_VAL) & (checked == False)
            )
            
            if unchecked_line_pixels.size == 0:
                break
            
            loop_count += 1
            if __debug__:
                print( f"--- Tracking (Phase {loop_count})" )
            
            # Get unchecked line pixel
            y, x = unchecked_line_pixels[0]
            
            # Init queue
            q = queue.Queue()
            q.put( (y, x) )
            
            # checked: (x, y)
            checked[y, x] = True
            
            # Loop until queue is empty
            while not q.empty():
                
                # deque
                y, x = q.get()
                
                if __debug__:
                    print( f"\rPixel: ({x}, {y}) ", end="", flush=True )
                
                # Set of neighbor region's label
                label_set = set()
                
                for d in D:
                    p = (y + d[0], x + d[1])
                    if 0 <= p[0] < labels.shape[0] and 0 <= p[1] < labels.shape[1]:
                        if labels[p] == self.LINE_LABEL_VAL:
                            if not checked[p]:
                                # Mark 'p' as 'checked'
                                checked[p] = True
                                # Add Tracking Queue
                                q.put( p )
                        else:
                            label_set.add( int( labels[p] ) )
                
                for label_a, label_b in [z for z in itertools.product( label_set, repeat=2 ) if z[0] != z[1]]:
                    if label_a not in relation:
                        relation[label_a] = dict()
                    if label_b not in relation[label_a]:
                        relation[label_a][label_b] = set()
                    
                    relation[label_a][label_b].add( (y, x) )
            
            print( "" )
        
        return relation
    
    @dec_debug
    def calc_region_pair( self ):
        relation = self.relations
        
        region_pair = set()
        
        for label_1, v in relation.items():
            for label_2, _ in v.items():
                region_pair.add( tuple( sorted( [label_1, label_2] ) ) )
        
        return region_pair
    
    @dec_debug
    def calc_score( self, label_1, label_2, distance=5 ):
        TYPE_ASSERT( label_1, int )
        TYPE_ASSERT( label_2, int )
        
        relation = self.relations
        img = self.src_img
        
        edge = EdgeProcedures( self.src_img )
        edge_angle = edge._calc_angle()
        
        if label_1 not in relation:
            raise KeyError( " Label '{label_1}' is not contain in 'self.relation'".format(
                label_1=label_1
            ) )
        if label_2 not in relation[label_1]:
            raise KeyError( " Label '{label_2}' is not contain in 'self.relation[{label_1}]'".format(
                label_1=label_1,
                label_2=label_2
            ) )
        
        connected_points = list( relation[label_1][label_2] )
        angle_of_points = np.array( [edge_angle[tuple( point )] for point in connected_points], dtype=np.float32 )
        
        pts_1, pts_2 = EdgeProcedures.calc_end_points(
            points=connected_points,
            deg_angles=angle_of_points,
            distance=distance
        )
        
        # score = 0
        #
        # for pt_1, pt_2 in zip( pts_1, pts_2 ):
        #     val_1 = cv2.cvtColor( img[tuple( pt_1 )].reshape( 1, 1, 3 ), cv2.COLOR_BGR2HSV )[0, 0].astype( np.int32 )
        #     val_2 = cv2.cvtColor( img[tuple( pt_2 )].reshape( 1, 1, 3 ), cv2.COLOR_BGR2HSV )[0, 0].astype( np.int32 )
        #
        #     score += np.abs( val_1 - val_2 )
        #
        # score = np.sum( score )
        
        
        pixels_1 = np.array( [img[tuple( point )] for point in pts_1], dtype=img.dtype )
        pixels_2 = np.array( [img[tuple( point )] for point in pts_2], dtype=img.dtype )
        
        score = np.mean(
            np.fabs(
                np.std( pixels_1, axis=0 ) - np.std( pixels_2, axis=0 )
            )
        )
        
        print( "Score[ Sum of Difference, Label: ({label_1}, {label_2}) ] = {score}".format(
            label_1=label_1,
            label_2=label_2,
            score=score
        ) )
        
        return score
    
    @dec_debug
    def calc_score_all( self, distance=5 ):
        relation = self.relations
        img = self.src_img
        
        edge = EdgeProcedures( self.src_img )
        edge_angle = edge._calc_angle()
        
        scores = dict()
        
        for label_1 in relation.keys():
            for label_2, connected_points in relation[label_1].items():
                
                eprint( "\rCalculating Score [ Label: ({label_1}, {label_2}) ] ...".format(
                    label_1=label_1,
                    label_2=label_2
                ), end="" )
                
                if label_1 not in scores:
                    scores[label_1] = dict()
                
                connected_points = list( connected_points )
                angle_of_points = np.array( [edge_angle[tuple( point )] for point in connected_points],
                                            dtype=np.float32 )
                
                pts_1, pts_2 = EdgeProcedures.calc_end_points(
                    points=connected_points,
                    deg_angles=angle_of_points,
                    distance=distance
                )
                
                pts = np.stack( [pts_1, pts_2], axis=1 )
                
                # Filtering if in range coordinates
                pts_filtered = pts[
                    ((0 <= pts[:, 0, 0]) & (pts[:, 0, 0] < img.shape[0])) &
                    ((0 <= pts[:, 0, 1]) & (pts[:, 0, 1] < img.shape[1])) &
                    ((0 <= pts[:, 1, 0]) & (pts[:, 1, 0] < img.shape[0])) &
                    ((0 <= pts[:, 1, 1]) & (pts[:, 1, 1] < img.shape[1]))
                    ]
                
                val = np.zeros( tuple( (*pts_filtered.shape[:2], 3) ), dtype=np.int32 )
                
                val[:, 0] = [
                    cv2.cvtColor( img[tuple( pt )].reshape( 1, 1, 3 ), cv2.COLOR_BGR2HSV )[0, 0].astype( np.int32 ) for
                    pt in pts_filtered[:, 0]]
                val[:, 1] = [
                    cv2.cvtColor( img[tuple( pt )].reshape( 1, 1, 3 ), cv2.COLOR_BGR2HSV )[0, 0].astype( np.int32 ) for
                    pt in pts_filtered[:, 1]]
                
                # for pt_1, pt_2 in zip( pts_1, pts_2 ):
                #     val_1 = cv2.cvtColor( img[tuple( pt_1 )].reshape( 1, 1, 3 ), cv2.COLOR_BGR2HSV )[0, 0].astype( np.int32 )
                #     val_2 = cv2.cvtColor( img[tuple( pt_2 )].reshape( 1, 1, 3 ), cv2.COLOR_BGR2HSV )[0, 0].astype( np.int32 )
                #
                #     average_of_difference += np.abs( val_1 - val_2 )
                
                scores[label_1][label_2] = np.mean( np.sum( np.abs( val[:, 0] - val[:, 1] ), axis=1 ) )
                
                # return cv2.cvtColor( img[tuple(x)].reshape( 1, 1, 3 ), cv2.COLOR_BGR2HSV )[0, 0].astype( np.int32 )
        return scores
    
    @dec_debug
    def calc_entropy( self, label_1, label_2, mode='rgb' ):
        """
        2 ラベル間のエントロピーを計算
            - 各領域と、領域統合後のエントロピー
              の差分を計算する

        Parameters
        ----------
        label_1, label_2 : int
            ラベル番号
        mode : str
            エントロピーを計算する色空間を指定
            'hsv'  : HSV 色空間で計算
            その他 : RGB 色空間で計算

        Returns
        -------
        np.float64
            label_1 と label_1 & label_2 間、
            label_2 と label_1 & label_2 間の
            エントロピー差分のうち、より小さい方の
            値

        """
        TYPE_ASSERT( label_1, int )
        TYPE_ASSERT( label_2, int )
        TYPE_ASSERT( mode, str )
        
        src_img = self.src_img
        labels = self.labels
        
        if mode == 'hsv':
            src_img = cv2.cvtColor(
                self.src_img.copy(),
                cv2.COLOR_BGR2HSV
            )
        else:
            src_img = self.src_img
        
        a = np.mean(
            np.abs(
                entropy( src_img[labels == label_1] ) - entropy( src_img[(labels == label_1) | (labels == label_2)] )
            )
        )
        b = np.mean(
            np.abs(
                entropy( src_img[labels == label_2] ) - entropy( src_img[(labels == label_1) | (labels == label_2)] )
            )
        )
        
        return np.min( [a, b] )
    
    @dec_debug
    def calc_entropy_all( self, mode='rgb' ):
        """
        隣接領域間のエントロピーを計算
            - 各領域と、領域統合後のエントロピーの
              差分を計算する

        Parameters
        ----------
        mode : str
            エントロピーを計算する色空間を指定
            'hsv'  : HSV 色空間で計算
            その他 : RGB 色空間で計算

        Returns
        -------
        np.float64
            エントロピー差分のうち、より小さい方の値

        """
        TYPE_ASSERT( mode, str )
        
        src_img = self.src_img
        labels = self.labels
        
        scores = dict()
        
        if mode == 'hsv':
            src_img = cv2.cvtColor(
                self.src_img.copy(),
                cv2.COLOR_BGR2HSV
            )
        else:
            src_img = self.src_img
        
        for label_1, label_2 in self.calc_region_pair():
            a = np.mean(
                np.abs(
                    entropy( src_img[labels == label_1] ) - entropy(
                        src_img[(labels == label_1) | (labels == label_2)] )
                )
            )
            b = np.mean(
                np.abs(
                    entropy( src_img[labels == label_2] ) - entropy(
                        src_img[(labels == label_1) | (labels == label_2)] )
                )
            )
            score = np.min( [a, b] )
            
            if label_1 not in scores:
                scores[label_1] = dict()
            
            scores[label_1][label_2] = score
        
        return scores
    
    @dec_debug
    def merge_region( self, label_1, label_2 ):
        """
        label_1, label_2の領域統合を行う
          - label_1, label_2 のうち、領域の
            大きい方に統合する
          - label_1 と label_2 の境界線を
            領域の大きい方に統合する
        
        Parameters
        ----------
        label_1, label_2 : int
            ラベル番号

        Returns
        -------
        """
        TYPE_ASSERT( label_1, [int, np.sctypes['int'], np.sctypes['uint']] )
        TYPE_ASSERT( label_2, [int, np.sctypes['int'], np.sctypes['uint']] )
        
        merged_labels = self.merged_labels
        relation = self.relations
        lut = self.labels_lut
        
        lut_label_1, lut_label_2 = lut[label_1], lut[label_2]
        
        # Calc area
        area_1 = merged_labels[merged_labels == lut_label_1].size
        area_2 = merged_labels[merged_labels == lut_label_2].size
        
        # 領域の面積によって、大小を決定する
        larger_label, smaller_label = (lut_label_1, lut_label_2) if area_1 >= area_2 else (lut_label_2, lut_label_1)
        
        ##### 領域の統合 #####
        
        
        # Merge smaller area into larger area
        merged_labels[merged_labels == smaller_label] = larger_label
        
        ###### 線の消去 ######
        for border_point in relation[label_1][label_2]:
            merged_labels[tuple( border_point )] = larger_label
        
        # Update Look-Up-Table
        lut[smaller_label] = larger_label
        
        return
    
    @dec_debug
    def merge_regions_by_score( self, entropy_calc_mode='rgb', condition_to_merge="score < 0.5" ):
        """
        エントロピーに基づいた領域統合
          - 隣接領域間のエントロピー差分 (score) を計算
          - score が 引数 condition_to_merge の条件を
            満たす場合、領域統合が行われる
            
        Parameters
        ----------
        entropy_calc_mode : str
            関数 calc_entropy_all の mode 引数に対応する
        
        condition_to_merge : str
            領域統合を行う条件式

        Returns
        -------
        """
        TYPE_ASSERT( entropy_calc_mode, str )
        TYPE_ASSERT( condition_to_merge, str )
        
        scores = self.calc_entropy_all( mode=entropy_calc_mode )
        
        try:
            score = 1.0
            eval( condition_to_merge )
        except Exception as e:
            eprint(
                "Catch exception while evaluation 'condition_to_merge'\ncondition_to_merge: '{condition_to_merge}'".format(
                    condition_to_merge=condition_to_merge
                ) )
            print( e )
        
        f = open( "merged_regions.csv", "wt" )
        
        for label_1, v in scores.items():
            for label_2, score in v.items():
                if eval( condition_to_merge ):
                    self.merge_region( label_1, label_2 )
        f.close()
        
        # Re-Labeling
        # merged_labels = self.merged_labels
        # merged_labels[merged_labels == self.LINE_LABEL_VAL] = self.LINE_PIXEL_VAL
        # merged_labels[merged_labels != self.LINE_PIXEL_VAL] = np.bitwise_not(self.LINE_PIXEL_VAL)
        
        # Opening lines
        # kernel = np.array([
        #     [1, 1, 1],
        #     [1, 1, 1],
        #     [1, 1, 1]
        # ], dtype=np.uint8)
        # merged_labels = cv2.morphologyEx(merged_labels.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # _, self.merged_labels = cv2.connectedComponents(
        #     np.bitwise_not( merged_labels.astype(np.uint8) ),
        #     connectivity=4
        # )
        
        
        if self.is_logging:
            self.logger.logging_img( self._labels, "labels" )
        if self.is_logging:
            self.logger.logging_img( self._merged_labels, "labels_merged" )
    
    @dec_debug
    def point_interpolation( self ):
        """
        領域統合により途切れてしまった
        領域分割線を補完する

        補完は次のアルゴリズムで行われる
        1. 線分追跡を行い、「端点」の検出
           を行う
        2. 各端点を中心に 5x5 の範囲に端点
           があるか探索する
        3. 端点が見つかった場合、2端点間を
           補完する

        Returns
        -------
        """
        from itertools import chain
        
        # 「線分画素」の画素値
        LINE_PIX_VAL = 255
        
        # 端点テンプレート (『ディジタル信号処理』 p.194)
        T_ENDPOINTS = list( chain.from_iterable( [
            [np.rot90( np.array( arr, dtype=np.int32 ) * LINE_PIX_VAL, k ) for k in range( 4 )]
            for arr in [
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [-1, 1, -1]
                ],
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [1, 0, 0]
                ],
            ]
        ] ) )
        
        def is_endpoints( i, j ):
            """ Img(j, i) が端点かどうか判定するサブルーチン """
            
            # (範囲チェック)
            if 1 <= i <= line_img.shape[0] - 2 and 1 <= j <= line_img.shape[1] - 2:
                # 中心 (j, i) から 3x3 の範囲を切り取り、
                roi = line_img[i - 1:i + 2, j - 1:j + 2]
                
                # 端点テンプレートにマッチするかどうかチェック。
                for template in T_ENDPOINTS:
                    if np.all( (roi == template)[template != -LINE_PIX_VAL] ):
                        return True
            
            return False
        
        # 領域統合処理後のラベリング結果から、2値化により領域分割線画像を生成
        line_img = self.binalization_labels( self.merged_labels )
        
        # 領域分割線画像をネガポジ反転
        line_img = np.bitwise_not( line_img )
        
        # checked: 探索済みの座標を格納
        checked = set()
        # connected_pair: 補完が必要な二点間の端点座標のペアを格納
        connected_pair = list()
        
        # メインルーチン
        for i in range( 1, line_img.shape[0] - 1 ):
            for j in range( 1, line_img.shape[1] - 1 ):
                
                # もし注目画素が「線分画素」で、
                if line_img[i, j] == LINE_PIX_VAL:
                    
                    # その画素が「探索済み」でなく、「端点」であった場合、
                    if (i, j) not in checked and is_endpoints( i, j ):
                        
                        # その画素を探索済みとする。
                        checked.add( (i, j) )
                        
                        # (範囲チェック)
                        if 2 <= i < line_img.shape[0] - 2 and 2 <= j < line_img.shape[1] - 2:
                            
                            # そして、中心画素から 5x5 を切り取り
                            roi = line_img[i - 2:i + 3, j - 2:j + 3]
                            
                            # その中に
                            for k in range( roi.shape[0] ):
                                for l in range( roi.shape[1] ):
                                    
                                    p = (i + (k - 2), j + (l - 2))
                                    
                                    # 端点画素があった場合
                                    if p != (i, j) and is_endpoints( *p ):
                                        
                                        # (i, j) と (i + (k-2), j + (l-2)) の2点を
                                        # 補完処理リストに追加する。
                                        connected_pair.append( ((i, j), p) )
                                        checked.add( p )
        
        # 補完処理リストを処理
        for ((y_1, x_1), (y_2, x_2)) in connected_pair:
            
            # center: 二点間の中点
            center = ((y_1 + y_2) // 2, (x_1 + x_2) // 2)
            
            # 座標 center の画素を「線分画素」に
            line_img[center] = LINE_PIX_VAL
        
        if self.is_logging:
            self.logger.logging_img( line_img, "point_interpolated" )
        
        return line_img

    def do_segmentation( self, method="watershed" ):
        TYPE_ASSERT( method, str )

        if method == "watershed":
            return self._watershed()
        elif method == "watershed_quasi_dist":
            return self._watershed_using_quasi_distance()
        else:
            raise RuntimeError(
                "Not support method: {method}".format(
                    method=method
                )
            )
    
    # Constructor
    def __init__( self, img, logging=False,
                  logging_base_path="./img/tmp/binary_segmentation" ) -> None:
        """
                コンストラクタ

                Parameters
                ----------
                img : numpy.ndarray or str
                    エッジ抽出を行う入力画像
                    画像データ(numpy.ndarray)と
                    画像ファイルへのパス(str)の
                    両方が許容される
                """
        
        TYPE_ASSERT( img, (str, np.ndarray) )
        TYPE_ASSERT( logging, bool, allow_empty=True )
        
        self.src_img = None
        self._segmented_line_img = None
        self.n_labels = -1
        self._labels = None
        self._merged_labels = None
        self._labels_lut = None
        self._relations = None
        self._scores = None
        self.logger = None
        
        if isinstance( img, str ):
            if not path.exists( img ):
                raise InvalidImageOrFile( "Cannot find file -- '{path}'".format( path=img ) )
            else:
                self.src_img = cv2.imread( img, cv2.IMREAD_COLOR )
        
        elif isinstance( img, np.ndarray ):
            if img.ndim == 2:
                self.src_img = cv2.cvtColor( img, cv2.COLOR_GRAY2BGR )
            else:
                self.src_img = img.copy()
        
        if logging:
            self.logger = ImageLogger( base_path=logging_base_path )
    
    
    # Properties
    @property
    def is_logging( self ):
        return self.logger is not None
    
    @property
    def segmented_line_img( self ):
        """
        プロパティ : segmented_line_img
            領域分割線画像
            背景が黒(0)、線が白(255)
            
        Returns
        -------

        """
        if self._segmented_line_img is None:
            self.segmented_line_img = self.watershed_using_quasi_distance()
        
        return self._segmented_line_img
    
    @segmented_line_img.setter
    def segmented_line_img( self, value ):
        self._segmented_line_img = value
        
        if self.is_logging:
            self.logger.logging_img( self._segmented_line_img, "segmented_lines" )
    
    
    @property
    def labels( self ):
        if self._labels is None:
            self.labels = self.labeling_by_segmented_img()
        
        return self._labels
    
    @labels.setter
    def labels( self, value ):
        self._labels = value
        
        if self.is_logging:
            self.logger.logging_img( self._labels, "labels" )
    
    @property
    def merged_labels( self ):
        if self._merged_labels is None:
            self.merged_labels = self.labels.copy()
        
        return self._merged_labels
    
    @merged_labels.setter
    def merged_labels( self, value ):
        self._merged_labels = value
        
        if self.is_logging:
            self.logger.logging_img( self._merged_labels, "labels_merged" )
    
    @property
    def labels_lut( self ):
        if self._labels_lut is None:
            self.labels_lut = { k: k for k in range( self.n_labels ) }
        
        return self._labels_lut
    
    @labels_lut.setter
    def labels_lut( self, value ):
        self._labels_lut = value
        
        if self.is_logging:
            self.logger.logging_dict( self._labels_lut, "labels_lut" )
    
    
    @property
    def relations( self ):
        if self._relations is None:
            self.relations = self.calc_region_connection()
        
        return self._relations
    
    @relations.setter
    def relations( self, value ):
        self._relations = value
        
        if self.is_logging:
            self.logger.logging_dict( self._relations, "relations" )
    
    @property
    def scores( self ):
        if self._scores is None:
            self.scores = self.calc_score_all()
        
        return self._scores
    
    @scores.setter
    def scores( self, value ):
        self._scores = value
        
        if self.is_logging:
            self.logger.logging_dict( self._scores, "scores" )
