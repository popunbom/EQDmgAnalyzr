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
    LINE_PIXEL_VAL = 255
    LINE_LABEL_VAL = 0
    
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
    
    def labeling_by_segmented_img( self ):
        """
        
        領域分割線画像をラベリング
        
        Returns
        -------
        numpy.ndarray
            ラベリング結果(dtype=np.int)
        """
        
        self.n_labels, labels = cv2.connectedComponents(
            np.bitwise_not( self.segmented_line_img ),
            connectivity=4
        )
        
        return labels
    
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
    
    def calc_entropy( self, label_1, label_2, mode='rgb' ):
        """
        2 ラベル間のエントロピーを計算
            各2領域と、領域統合後のエントロピーの差分
            を計算する
        
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
        
        return np.min([a, b])
    
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
        self._labels = None
        self._relations = None
        self._scores = None
        self.logger = None
        self.n_labels = -1
        
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
