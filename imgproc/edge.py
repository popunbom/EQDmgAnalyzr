# /usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: popunbom <fantom0779@gmail.com>
# Created At: 2019/06/21

# エッジ処理に係る処理

from os import path

import cv2
import numpy as np

from skimage.filters import sobel

from imgproc.utils import compute_by_window
from utils.assertion import TYPE_ASSERT, NDIM_ASSERT, SAME_SHAPE_ASSERT
from utils.exception import InvalidImageOrFile


class EdgeProcedures( object ):
    """
    エッジ抽出に関わる処理を行う
    
    Attributes
    ----------
    src_img : numpy.ndarray
        エッジ抽出を行う入力画像 (Grayscale)
    detector_params : dict
        エッジ検出のパラメータ
    edge_magnitude : numpy.ndarray (dtype=np.float32)
        エッジ強度 : [0, 1.0]
    edge_angle : numpy.ndarray (dtype=np.float32)
        エッジ強度 : [0°, 360°]
    """
    
    @staticmethod
    def calc_end_points( points, deg_angles, distance ):
        """
        中心座標を(x, y)、角度を deg_angle、2点間の距離を
        distance としたときの、両端点の座標を計算する
        
        Parameters
        ----------
        points : list of list of int or numpy.ndarray
            中心座標(Y, X)
        deg_angles : list of int or numpy.ndarray
            線分の傾きの角度(度数法)
        distance : int or numpy.int or numpy.uint
            2点間の距離

        Returns
        -------
        pts_1, pts_2 : numpy.ndarray
            両端点の座標(Y, X)
        """
        
        # convert Degree to Radian
        rad_angles = np.radians( deg_angles )
        
        # P: Base point, A: Angle, D: Difference
        P = np.array( points, dtype=np.float32 )
        A = np.stack( [np.sin( rad_angles ), np.cos( rad_angles )], axis=1 )
        D = distance / 2
        
        pts_1, pts_2 = P + (A * D), P - (A * D)
        
        pts_1 = pts_1.astype( np.int16 )
        pts_2 = pts_2.astype( np.int16 )
        
        return pts_1, pts_2
    
    @staticmethod
    def angle_variance_using_mode( _edge_magnitude, _edge_angle ):
        """
        最頻角度に基づくエッジ角度分散の計算
            通常の分散の計算では平均値を用いるが、角度の
            平均値は単純に求めることが難しいため、最頻値
            となる角度の全体の角度に対する割合を擬似的に
            エッジ角度分散とする (星野, 2016)
            
        Parameters
        ----------
        _edge_magnitude : numpy.ndarray
            エッジ強度
        _edge_angle : numpy.ndarray
            エッジ角度

        Returns
        -------
        float
            エッジ角度の分散値 [0, 1]
            

        """
        magnitude = _edge_magnitude
        angle = _edge_angle
        
        hist, _ = np.histogram(
            angle,
            weights=magnitude,
            bins=list( range( 361 ) )
        )
        
        variance = np.max( hist ) / np.sum( magnitude )
        
        return variance
    
    @staticmethod
    def angle_variance_using_mean_vector( _edge_magnitude, _edge_angle ):
        """
        平均ベクトルに基づくエッジ角度分散の計算
            *REF: [\[PDF\] 角度統計](http://q-bio.jp/images/5/53/角度統計配布_qbio4th.pdf)*
        
        - $N$ : 計算対象の角度の個数
        - まず、$\cos$、$\sin$ の平均値を計算する
            - それぞれを $M_{\cos}$、$M_{\sin}$ とすると
            $$ M_{\cos} = \frac{1}{N} \sum^{N} \cos{\theta} \;, \;\; M_{\sin} = \frac{1}{N} \sum^{N} \sin{\theta} $$
            
        - ここで、平均ベクトルを考える
            - 平均ベクトル $(R\cos{\Theta}, R\sin{\Theta})$ は次のように定義される
                - $R$: ベクトルの長さ
            $$ (R\cos{\Theta}, R\sin{\Theta}) = (M_{\cos}, M_{\sin}) $$
        
        - このとき、エッジ角度分散 $V$ は平均ベクトルの長さ $R$ を用いて次のように定義される
            $$ V = 1 - R $$
            - $R$ は以下の計算で算出する
            $$ R = \sqrt{ {M_{\cos}}^2 + {M_{\sin}}^2 } $$
        Parameters
        ----------
        _edge_magnitude : numpy.ndarray
            エッジ強度
        _edge_angle : numpy.ndarray
            エッジ角度

        Returns
        -------
        float
            エッジ角度の分散値 [0, 1]

        """
        degrees = _edge_angle
        weights = _edge_magnitude

        if np.isclose( np.sum( weights ), 0 ):
            return 0

        # weights /= weights.max()
        
        M_cos = np.average(
            np.cos( np.radians( degrees ) ),
            weights=weights
        )
        M_sin = np.average(
            np.sin( np.radians( degrees ) ),
            weights=weights
        )

        # R = np.sqrt( M_cos ** 2 + M_sin ** 2 )
        R = np.hypot( M_cos, M_sin )
        
        variance = 1 - R
        
        return variance
    
    @staticmethod
    def magnitude_stddev( _edge_magnitude, _edge_angle ):
        return np.std( _edge_magnitude )
    
    def set_detector_params( self, **kwargs ):
        """
        エッジ検出の際のパラメーター設定
        Parameters
        ----------
        **kwargs
            設定したいパラメータ値
            キーワード引数として渡す
            
        Returns
        -------
        dict
            設定反映後のパラメーター一覧

        """
        self.detector_params.update( kwargs )
        
        return self.detector_params
    
    def get_magnitude( self ):
        """
        Sobel フィルタによるエッジ強度の抽出
        
        Returns
        -------
        numpy.ndarray
            エッジ強度 (dtype=np.float32, [0, 1.0])
        
    
        """

        print( "Sobel Edge Detector using Scikit-Image -- with mean" )
        # Sobel Edge Detector
        # dx = cv2.Sobel( self.src_img, cv2.CV_32F, 1, 0, ksize=self.detector_params['ksize'] )
        # dy = cv2.Sobel( self.src_img, cv2.CV_32F, 0, 1, ksize=self.detector_params['ksize'] )
        
        # Edge Magnitude (Normalize to [0, 1.0])
        # magnitude = np.sqrt( dx * dx + dy * dy )
        # magnitude /= magnitude.max()

        magnitude = sobel( self.src_img )
        
        return magnitude
    
    def get_angle( self ):
        """
        Sobel フィルタによるエッジ強度の抽出
        
        Returns
        -------
        numpy.ndarray
            エッジ強度(dtype=np.float32, [0°, 360°])
        """
        
        # Sobel Edge Detector
        dx = cv2.Sobel( self.src_img, cv2.CV_32F, 1, 0, ksize=self.detector_params['ksize'] )
        dy = cv2.Sobel( self.src_img, cv2.CV_32F, 0, 1, ksize=self.detector_params['ksize'] )

        # Edge Angle (0° <= θ <= 360°)
        angle = np.round( np.degrees( np.arctan2( dy, dx ) ) ) + 180
        
        return angle
    
    def get_feature_by_window( self, func, window_size, step ):
        """
        ウィンドウで切りながら、エッジ特徴量を計算
        
        Parameters
        ----------
        func : callable object
            特徴量計算を行う関数
        window_size : int or tuple of int
            画像を切り取るサイズ
            詳細は SharedProcedures.compute_by_window を参照
        step : int or tuple of int
            切り取り間隔
            詳細は SharedProcedures.compute_by_window を参照

        Returns
        -------
        numpy.ndarray
            特徴量値
        """
        
        return compute_by_window(
            (self.edge_magnitude, self.edge_angle),
            func,
            window_size=window_size,
            step=step,
            dst_dtype=np.float64
        )

    def get_angle_colorized_img( self, normalized_magnitude=False, max_intensity=False, mask_img=None ):
        """
        エッジ角度の疑似カラー画像を生成
            Hue (色相) に角度値を割り当て、HSV→RGB への
            色空間変換を利用して疑似カラー画像生成を行う。


        Parameters
        ----------
        edge_angle : numpy.ndarray
            エッジ角度 : [0°, 360°]
        edge_magnitude : numpy.ndarray
            エッジ強度 : [0, 1.0]
        max_intensity : bool, default False
            True のとき、HSV の V 値は全画素において最大値となる
            False のとき、HSV の V 値には正規化されたエッジ強度値
            が割り当てられる
        mask_img : numpy.ndarray, default None
            マスク画像(2値化済み)
            白色(非ゼロ値)を透過対象とする
            mask_img が与えられた場合、疑似カラー画像に対して
            マスクを適用した結果が返却される


        Returns
        -------
        numpy.ndarray
            mask_img が None の場合、疑似カラー画像、mask_img が
            None でない場合、マスク済み疑似カラー画像が返却される
        """
        
        TYPE_ASSERT( mask_img, [None, np.ndarray] )
        if isinstance( mask_img, np.ndarray ):
            NDIM_ASSERT( mask_img, 2 )
        
        magnitude, angle = self.edge_magnitude, self.edge_angle
        
        hue = (angle / 2).astype( np.uint8 )

        saturation = np.ones( hue.shape, dtype=hue.dtype ) * 255

        if max_intensity:
            value = np.ones( hue.shape, dtype=hue.dtype ) * 255
        else:
            if normalized_magnitude:
                value = (magnitude * (255.0 / magnitude.max())).astype( np.uint8 )
            else:
                value = (magnitude * 255.0).astype( np.uint8 )
        
        angle_img = cv2.cvtColor( np.stack( [hue, saturation, value], axis=2 ), cv2.COLOR_HSV2BGR )
        
        if mask_img is None:
            return angle_img
        else:
            if mask_img.max() != 1:
                mask_img[mask_img > 0] = 1
            
            masked_img = angle_img * np.stack( [mask_img] * 3, axis=2 )
            
            return masked_img
    
    def draw_edge_angle_line( self, line_color=(255, 255, 255), line_length=10,
                              draw_on_angle_img=True, mask_img=None ):
        """
        エッジ角度に対応する線分を描画する

        Parameters
        ----------
        src_img : numpy.ndarray
            角度線が描画されるベース画像
        edge_angle : numpy.ndarray
            エッジ角度
        line_color : tuple
            線の色(R, G, B の順)
        line_length : int
            線の長さ
        mask_img : [None, numpy.ndarray]
            マスク画像(2値化済み)
            mask_img が与えられた場合、白色(非ゼロ値)の
            箇所のみ角度線が描画される

        Returns
        -------
        angle_line_img : numpy.ndarray
            角度線が描画された画像(BGR)
            線描画の都合上、画像の大きさが縦、横
            それぞれ3倍されて返却される
        """
        
        TYPE_ASSERT( mask_img, [None, np.ndarray] )
        NDIM_ASSERT( mask_img, 2 )
        
        if draw_on_angle_img:
            base_img = self.get_angle_colorized_img( max_intensity=True, mask_img=mask_img )
        else:
            base_img = self.src_img
        
        SAME_SHAPE_ASSERT( mask_img, base_img, ignore_ndim=True )
        
        angles = self.get_angle()
        
        angle_line_img = cv2.resize(
            base_img, dsize=None, fx=3.0, fy=3.0, interpolation=cv2.INTER_NEAREST
        )
        
        # Vectorization Process
        if mask_img is not None:
            draw_points = (np.argwhere( mask_img != 0 ) + 1) * 3 - 2
            angles = angles[mask_img != 0].flatten()
        else:
            draw_points = np.stack( np.meshgrid( *[np.arange( i ) for i in mask_img.shape[:2]] ), axis=2 )
            angles = angles.flatten()
        
        pts_1, pts_2 = self.calc_end_points( draw_points, angles, line_length )
        
        # Line Drawing
        for (i, (pt_1, pt_2)) in enumerate( zip( pts_1, pts_2 ) ):
            pt_1, pt_2 = tuple( pt_1[::-1] ), tuple( pt_2[::-1] )
            print( f"\rComputing ... [ {i} / {pts_1.shape[0]} ]", end="", flush=True )
            cv2.line( angle_line_img, pt_1, pt_2, line_color, thickness=1 )
        
        return angle_line_img
    
    
    # Constructor
    def __init__( self, img ) -> None:
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
        
        # TODO: cv2.cvtColor(BGR2GRAY) と imread(IMREAD_GRAYSCALE) の結果が異なる！？
        
        TYPE_ASSERT( img, (str, np.ndarray) )
        
        if isinstance( img, str ):
            if not path.exists( img ):
                raise InvalidImageOrFile( "Cannot find file -- '{path}'".format( path=img ) )
            else:
                self.src_img = cv2.imread( img, cv2.IMREAD_GRAYSCALE )
        
        elif isinstance( img, np.ndarray ):
            if img.ndim == 3:
                self.src_img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
            else:
                self.src_img = img.copy()
        
        self.detector_params = {
            'ksize': 3
        }
        
        self._edge_magnitude = None
        self._edge_angle = None
    
    @property
    def edge_magnitude( self ):
        if self._edge_magnitude is None:
            self.edge_magnitude = self.get_magnitude()
        
        return self._edge_magnitude
    
    @edge_magnitude.setter
    def edge_magnitude( self, value ):
        self._edge_magnitude = value
    
    
    @property
    def edge_angle( self ):
        if self._edge_angle is None:
            self.edge_angle = self.get_angle()
        
        return self._edge_angle
    
    @edge_angle.setter
    def edge_angle( self, value ):
        self._edge_angle = value
