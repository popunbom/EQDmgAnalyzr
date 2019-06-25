# /usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: popunbom <fantom0779@gmail.com>
# Created At: 2019/06/21


import os
import time
from datetime import datetime

import cv2
import numpy as np
from mamba import imageMb
from mamba.miscellaneous import Mamba2PIL, PIL2Mamba
from PIL import Image

from .SharedProcedures import *


class UnsupportedDataTypes( Exception ):
    
    
    def __init__( self, detail: str, **kwargs ) -> None:
        message = "Unsupported Data Types ({detail})".format(
            detail=detail
        )
        super().__init__( message )
        self.kwargs = kwargs


# PIL.Image.Image --> numpy.ndarray
def pil2cv( pil_img ):
    """
    PIL.Image.Image を numpy.ndarray に変換する
        REF: https://qiita.com/derodero24/items/f22c22b22451609908ee
    
    Parameters
    ----------
    pil_img : PIL.Image.Image
        変換元の PIL.Image.Image 画像データ

    Returns
    -------
    numpy.ndarray
        変換された numpy.ndarray 画像データ
    """
    TYPE_ASSERT( pil_img, Image.Image )
    
    npy_img = np.array( pil_img )
    
    # Data depth conversion
    if npy_img.dtype not in [np.uint8, np.float32]:
        npy_img = npy_img.astype( np.float32 )
        npy_img /= npy_img.max()
    
    if npy_img.ndim == 3:
        if npy_img.shape[2] == 3:
            # RGB -> BGR
            npy_img = cv2.cvtColor( npy_img, cv2.COLOR_RGB2BGR )
        elif npy_img.shape[2] == 4:
            # RGBA -> BGRA
            npy_img = cv2.cvtColor( npy_img, cv2.COLOR_RGBA2BGRA )
        else:
            raise UnsupportedDataTypes( "npy_img.shape = {shape}".format(
                shape=npy_img.shape
            ) )
    
    return npy_img


# numpy.ndarray --> mamba.base.imageMb
def cv2mamba( npy_img ):
    """
    numpy.ndarray を mamba.base.imageMb に変換する
    
    Parameters
    ----------
    npy_img : numpy.ndarray
        変換元の numpy.ndarray 画像データ

    Returns
    -------
    mamba.base.imageMb
        変換された mamba.base.imageMb 画像データ
    """
    
    TYPE_ASSERT( npy_img, np.ndarray )
    
    if npy_img.dtype == np.bool:
        bit_depth = 1
    elif npy_img.dtype == np.uint8:
        bit_depth = 8
    elif npy_img.dtype == np.float32:
        bit_depth = 32
    else:
        raise UnsupportedDataTypes( "npy_img.dtype = {dtype}".format(
            dtype=npy_img.dtype
        ) )
    
    mb_img = imageMb( npy_img.shape[1], npy_img.shape[0], bit_depth )
    
    if npy_img.ndim == 3:
        npy_img = cv2.cvtColor( npy_img, cv2.COLOR_BGR2RGB )
    
    PIL2Mamba( Image.fromarray( npy_img ), mb_img )
    
    return mb_img


# mamba.base.imageMb --> numpy.ndarray
def mamba2cv( mb_img ):
    """
    mamba.base.imageMb を numpy.ndarray に変換する
    
    Parameters
    ----------
    mb_img : mamba.base.imageMb
        変換元の mamba.base.imageMb 画像データ

    Returns
    -------
    numpy.ndarray
        変換された numpy.ndarray 画像データ
    """
    TYPE_ASSERT( mb_img, imageMb )
    
    return pil2cv( Mamba2PIL( mb_img ) )


class ImageLogger:
    """
    作業途中の画像をロギングするためのクラス
        各作業ディレクトリは、prefix, suffix, timestamp
        によって名前がつけられ、その中に画像が保存される
    
    Attributes
    ----------
    base_path : str
        ディレクトリを作成するルートパス
    suffix, prefix : str
        各ディレクトリ名の先頭(prefix)と末尾(suffix)
        につける文字列パターン
    separator : str
        prefix, suffix とtimestampを区切る文字列
    fmt_timestamp : str
        タイムスタンプ文字列のフォーマット文字列
    dir_name : str
        prefix, suffix, timestamp から構成される
        ディレクトリ名
    """
    
    def _update_timestamp( self ):
        """
        タイムスタンプを現在日時と時刻で更新する
        
        Returns
        -------
        """
        self.timestamp = datetime.now().strftime( self.fmt_timestamp )
        return

    def _logging_msg( self, *args, end="\n" ):
        msg = "::ImageLogging:: " + ' '.join( [str( e ) for e in args] ) + end
        sys.stderr.write( msg )
        sys.stderr.flush()
    
    def __init__( self, base_path, suffix="", prefix="", separator='_', fmt_timestamp="%Y%m%d_%H%M%S" ) -> None:
        """
        新たに ImageLogger インスタンスを作成する
        Parameters
        ----------
        base_path : str
            ディレクトリを作成するルートパス
        suffix, prefix : str
            各ディレクトリ名の先頭(prefix)と末尾(suffix)
            につける文字列パターン
        separator : str
            prefix, suffix とtimestampを区切る文字列
        fmt_timestamp : str
            タイムスタンプ文字列のフォーマット文字列
        """
        TYPE_ASSERT( base_path, str )
        TYPE_ASSERT( suffix, str, allow_empty=True )
        TYPE_ASSERT( prefix, str, allow_empty=True )
        TYPE_ASSERT( separator, str )
        TYPE_ASSERT( fmt_timestamp, str )
        
        self.base_path = os.path.abspath( base_path )
        self.suffix = suffix
        self.prefix = prefix
        self.separator = separator
        self.fmt_timestamp = fmt_timestamp
        self.timestamp = None
        
        if not os.path.exists( self.base_path ):
            os.makedirs( self.base_path )
            print( "Created directory for logging -- ", self.base_path )
        
        self.makedir()
    
    @property
    def dir_name( self ):
        """
        プロパティ : dir_name
        
        Returns
        -------
        str
            self.prefix, self.timestamp, self.suffix
            をこの順番に、self.separator で連結した
            文字列

        """
        tokens = [self.prefix, self.timestamp, self.suffix]
        return self.separator.join( [t for t in tokens if t] )
    
    @property
    def dir_path( self ):
        """
        プロパティ : dir_path
        
        Returns
        -------
        str
            self.base_path と self.dir_name を連結
            させた、ディレクトリのパス文字列

        """
        return os.path.join( self.base_path, self.dir_name )
    
    def makedir( self ):
        """
        ログを保存するディレクトリを作成する
            prefix, timestamp, suffix に基づいてディレクトリ名を決定し、
            self.base_path 直下にディレクトリを作成する
            すでにフォルダが存在する場合、FileExistsError 例外が発生
            する
            
        Returns
        -------
        bool
            os.makedir の戻り値

        """
        
        # Regenerate Timestamp
        self._update_timestamp()
        
        self._logging_msg( "created directory -- '{dir_path}'".format(
            dir_path=self.dir_path
        ) )
        
        return os.mkdir( self.dir_path )
    
    def get_psuedo_colors( self, size=1, range_h=(0, 180), range_s=255, range_v=(100, 255) ):
        TYPE_ASSERT( size, [int, np.sctypes['uint'], np.sctypes['int']] )
        TYPE_ASSERT( range_h, [tuple, int, np.sctypes['uint'], np.sctypes['int']] )
        TYPE_ASSERT( range_s, [tuple, int, np.sctypes['uint'], np.sctypes['int']] )
        TYPE_ASSERT( range_v, [tuple, int, np.sctypes['uint'], np.sctypes['int']] )
        
        hsv = list()
        np.random.seed(int(time.time()))
        
        for _range in[range_h, range_s, range_v]:
            if isinstance(_range, tuple):
                hsv.append(np.random.randint(low=_range[0], high=_range[1], size=size, dtype=np.uint8))
            else:
                hsv.append(np.ones( (size), dtype=np.uint8 ) * _range)
        
        hsv = np.stack(hsv, axis=1).reshape(size, 1, 3)
        
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return rgb.reshape(size, 3)
    
    def get_psuedo_color_img( self, img ):
        TYPE_ASSERT( img, np.ndarray )
        
        
        psuedo_color_img = np.zeros( (*img.shape[:2], 3), dtype=np.uint8)
        
        max = np.max(img)
        
        colors = self.get_psuedo_colors(size=max, range_s=(100, 255), range_v=(100, 255))
        
        for label_num, color in enumerate(colors):
            psuedo_color_img[img == label_num] = color
            
        return psuedo_color_img
        
        
        
    def logging_img( self, img, file_name, overwrite=False, do_pseudo_color=False ):
        """
        画像をロギングする処理
        
        Parameters
        ----------
        img : numpy.ndarray or mamba.base.imageMb or PIL.Image.Image
            保存するための画像データ
            img は 8-Bit BGR, Grayscale 画像 または、
            32-Bit Grayscale 画像が想定されている
            保存形式は、8-Bit 画像の場合は PNG、その他の
            場合は TIFF 形式で保存される
        file_name : str
            画像データのファイル名
            name には拡張子を含んでいた場合でも、データ形式
            によって拡張子が PNG または TIFF に変換される
            (詳細は img 引数の説明に記載)
            
        overwrite : bool
            すでに name で指定された画像が存在していた場合に
            上書きをするかどうかのフラグ

        Returns
        -------

        """
        TYPE_ASSERT( img, [np.ndarray, Image.Image, imageMb] )
        TYPE_ASSERT( file_name, str )
        TYPE_ASSERT( overwrite, bool, allow_empty=True )
        
        # Generate file path
        save_path = os.path.join( self.dir_path, file_name )
        # TODO: 拡張子ありの際の Warning を出す？
        file_name = os.path.splitext( file_name )[0]
        
        # Check file existance
        if os.path.exists( save_path ):
            if overwrite:
                os.remove( save_path )
            else:
                raise FileExistsError( "Logging image already exists ! [name='{name}', path='{path}']".format(
                    name=file_name,
                    path=self.dir_path
                ) )
        
        # Data conversion (PIL.Image, imageMb --> numpy.ndarray)
        if isinstance( img, Image.Image ):
            img = pil2cv( img )
        elif isinstance( img, imageMb ):
            img = mamba2cv( img )
        else:
            # Pseudo colorization
            if do_pseudo_color:
                img = self.get_psuedo_color_img(img)
            
            # Convert data depth
            if img.dtype not in [np.uint8, np.float32]:
                img = img.astype( np.float32 )
                img /= img.max()
        
        # Write file
        if img.dtype != np.uint8:
            img_file_path = os.path.join( self.dir_path, file_name + ".tiff" )
        else:
            img_file_path = os.path.join( self.dir_path, file_name + ".png" )
            
        if cv2.imwrite( img_file_path, img ):
            self._logging_msg("logging image succesfully ! -- {img_file_path}".format(
                img_file_path=img_file_path
            ))
            
        return
    
    def logging_json( self, dict_obj, file_name, overwrite=False ):
        """
        dict を JSON としてロギング

        Parameters
        ----------
        dict_obj : dict
            JSON として保存する dict
            dict の value に json.dump として保存できない値を
            格納している場合、関数内関数の _jsonize
            に変更を加えることで対応できる
        file_name : str
            ファイル名
            拡張子を含んでいた場合でも ".json" に変換
            される

        overwrite : bool
            すでに file_name で指定されたファイルが存在
            していた場合に上書きをするかどうかのフラグ

        Returns
        -------

        """
        
        def _jsonize( obj ):
            """ オブジェクトを JSON が許容できる形式に変換する """
            if isinstance( obj, set ):
                """ imgproc.segmentation.RegionSegmentation.calc_region_connection 用の処理 """
                if all( [isinstance( elem, tuple ) and len( elem ) == 2 for elem in obj] ):
                    return sorted( list( obj ), key=lambda e: e[0] )
                
                else:
                    return list( obj )
            
            elif isinstance( obj, tuple( np.sctypes['int'] + np.sctypes['uint'] + np.sctypes['float'] ) ):
                return obj.astype( 'object' )
                # return obj.item()
            
            else:
                raise TypeError(
                    "'{obj_class}' is not support for Serialize at '_jsonize'".format(
                        obj_class=type( obj )
                    )
                )
        
        TYPE_ASSERT( dict_obj, dict, allow_empty=True )
        TYPE_ASSERT( file_name, str )
        TYPE_ASSERT( overwrite, bool, allow_empty=True )
        
        import json
        
        file_name = os.path.splitext( file_name )[0] + ".json"
        file_path = os.path.join( self.dir_path, file_name )
        
        if not overwrite and os.path.exists( file_path ):
            raise FileExistsError( "JSON file already exists ! -- '{file_path}'".format(
                file_path=file_path
            ) )
        
        with open( file_path, "wt" ) as f:
            result = json.dump( dict_obj,
                              f,
                              ensure_ascii=False,
                              sort_keys=True,
                              indent="\t",
                              default=_jsonize
                              )
        
        if result:
            self._logging_msg("logging json successfully -- {json_file_path}".format(
                json_file_path=file_path
            ))

        return
