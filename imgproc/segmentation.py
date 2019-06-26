# /usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: popunbom <fantom0779@gmail.com>
# Created At: 2019/06/21

from mamba import (
    imageMb, gradient, add, negate,
    quasiDistance, copyBytePlane,
    subConst, build, maxima, label,
    watershedSegment, logic, mix,
)
from os import path
import cv2
import numpy as np
from scipy.stats import entropy

from .SharedProcedures import *
from .Utils import ImageLogger, mamba2cv, pil2cv
from .Utils import cv2mamba
from .edge import EdgeProcedures


class RegionSegmentation:
    LINE_PIXEL_VAL = np.uint8( 255 )
    LINE_LABEL_VAL = np.int16( 0 )
    
    
    @dec_debug
    def watershed_using_quasi_distance( self ):
        """
        疑似ユークリッド距離(Quasi Distance) に基づく
        Watershed 領域分割
        
        Parameters
        ----------
        logging : bool
            作業途中の画像を保存するかどうかのフラグ

        Returns
        -------
        imWts : numpy.ndarray
            領域分割線の画像
        pilim : numpy.ndarray
            入力画像に領域分割線が描画された画像
        """
        
        # Channel Split
        if self.src_img.ndim == 3:
            imBlue, imGreen, imRed = [cv2mamba( self.src_img[:, :, i] ) for i in range( 3 )]
        elif self.src_img.ndim == 2:
            imBlue, imGreen, imRed = [cv2mamba( self.src_img )] * 3
        
        # We will perform a thick gradient on each color channel (contours in original
        # picture are more or less fuzzy) and we add all these gradients
        gradIm = imageMb( imRed )
        imWrk1 = imageMb( imRed )
        gradIm.reset()
        gradient( imRed, imWrk1, 2 )
        add( imWrk1, gradIm, gradIm )
        gradient( imGreen, imWrk1, 2 )
        add( imWrk1, gradIm, gradIm )
        gradient( imBlue, imWrk1, 2 )
        add( imWrk1, gradIm, gradIm )
        
        # Then we invert the gradient image and we compute its quasi-distance
        qDist = imageMb( gradIm, 32 )
        negate( gradIm, gradIm )
        quasiDistance( gradIm, imWrk1, qDist )
        
        if self.is_logging:
            self.logger.logging_img( imWrk1, "quasi_dist_gradient" )
            self.logger.logging_img( qDist, "quasi_dist" )
        
        # The maxima of the quasi-distance are extracted and filtered (too close maxima,
        # less than 6 pixels apart, are merged)
        imWrk2 = imageMb( imRed )
        imMark = imageMb( gradIm, 1 )
        copyBytePlane( qDist, 0, imWrk1 )
        subConst( imWrk1, 3, imWrk2 )
        build( imWrk1, imWrk2 )
        maxima( imWrk2, imMark )
        
        # The marker-controlled watershed of the gradient is performed
        imWts = imageMb( gradIm )
        label( imMark, qDist )
        negate( gradIm, gradIm )
        watershedSegment( gradIm, qDist )
        copyBytePlane( qDist, 3, imWts )
        
        # The segmented binary and color image are stored
        logic( imRed, imWts, imRed, "sup" )
        logic( imGreen, imWts, imGreen, "sup" )
        logic( imBlue, imWts, imBlue, "sup" )
        
        pilim = mix( imRed, imGreen, imBlue )
        
        if self.is_logging:
            self.logger.logging_img( pilim, "segmented_image" )
        
        imWts = mamba2cv( imWts )
        
        return imWts
    
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
        
        img_with_label = cv2.resize(
            self.segmented_line_img,
            dsize=None,
            fx=3.0,
            fy=3.0,
            interpolation=cv2.INTER_NEAREST
        )
        
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            np.bitwise_not( img_with_label ), connectivity=4
        )
        
        img_with_label = cv2.cvtColor( img_with_label, cv2.COLOR_GRAY2BGR )
        
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
        edge_angle = edge.get_angle()
        
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
        
        sum_of_difference = 0
        
        for pt_1, pt_2 in zip( pts_1, pts_2 ):
            val_1 = cv2.cvtColor( img[tuple( pt_1 )].reshape( 1, 1, 3 ), cv2.COLOR_BGR2HSV )[0, 0].astype( np.int32 )
            val_2 = cv2.cvtColor( img[tuple( pt_2 )].reshape( 1, 1, 3 ), cv2.COLOR_BGR2HSV )[0, 0].astype( np.int32 )
            
            sum_of_difference += np.abs( val_1 - val_2 )
        
        sum_of_difference = np.sum( sum_of_difference )
        
        print( "Score[ Sum of Difference, Label: ({label_1}, {label_2}) ] = {score}".format(
            label_1=label_1,
            label_2=label_2,
            score=sum_of_difference
        ) )
        
        return sum_of_difference
    
    @dec_debug
    def calc_score_all( self, distance=5 ):
        relation = self.relations
        img = self.src_img
        
        edge = EdgeProcedures( self.src_img )
        edge_angle = edge.get_angle()
        
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
                    f.write( f"{label_1}, {label_2}\n" )
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
        self.merged_labels = self.merged_labels

        # _, self.merged_labels = cv2.connectedComponents(
        #     np.bitwise_not( merged_labels.astype(np.uint8) ),
        #     connectivity=4
        # )
        
        
        if self.is_logging:
            self.logger.logging_img( self._labels, "labels" )
        if self.is_logging:
            self.logger.logging_img( self._merged_labels, "labels_merged" )
    
    
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
            self.logger.logging_json( self._labels_lut, "labels_lut" )
    
    
    @property
    def relations( self ):
        if self._relations is None:
            self.relations = self.calc_region_connection()
        
        return self._relations
    
    @relations.setter
    def relations( self, value ):
        self._relations = value
        
        if self.is_logging:
            self.logger.logging_json( self._relations, "relations" )
    
    @property
    def scores( self ):
        if self._scores is None:
            self.scores = self.calc_score_all()
        
        return self._scores
    
    @scores.setter
    def scores( self, value ):
        self._scores = value
        
        if self.is_logging:
            self.logger.logging_json( self._scores, "scores" )
