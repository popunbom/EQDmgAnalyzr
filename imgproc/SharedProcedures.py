# /usr/bin/env python3
# -*- coding: utf-8 -*-

# SharedProcedures : 各モジュールで共通する処理
# Author: popunbom <fantom0779@gmail.com>
# Created At: 2019/06/21


from collections.abc import Iterable as _Iterable
from numbers import Number
import sys
import inspect

import numpy as np

from math import ceil


def eprint( *args, end="\n" ):
    """
    print 関数の標準エラー出力版
    Parameters
    ----------
    args : list
        出力したい変数
    end : str, default "\n"
        行末に付加される文字列

    Returns
    -------
    """
    msg = ' '.join( [str( e ) for e in args] ) + end
    sys.stderr.write( msg )
    sys.stderr.flush()


class CannotFindVariable( Exception ):
    
    
    def __init__( self, *args: object ) -> None:
        super().__init__( *args )


class InvalidImageOrFile( Exception ):
    
    
    def __init__( self, *args: object ) -> None:
        super().__init__( *args )


def _get_var_name( var ):
    """
    変数から変数名を取得する
    
    Parameters
    ----------
    var : object
        変数名を取得したい変数

    Returns
    -------
    str
        取得した変数名
    """
    
    if var is None:
        return "None"
    
    for fi in reversed( inspect.stack() ):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len( names ) > 0:
            return "'" + names[0] + "'"
    else:
        raise CannotFindVariable( "Cannot find variable name" )


def _get_qualified_class_name( _type ):
    """
    オブジェクトの完全クラス名の取得
    
    Parameters
    ----------
    _type : type object
        取得したいオブジェクトの type オブジェクト

    Returns
    -------
    str
        オブジェクトの完全名
    """
    
    if _type is None:
        return "None"
    
    is_not_builtins = hasattr( _type, "__module__" ) and _type.__module__ != "builtins"
    
    if is_not_builtins:
        return "'{module_name}.{class_name}'".format(
            module_name=_type.__module__,
            class_name=_type.__name__
        )
    else:
        return _type.__name__


def _IS_EMPTY( var ):
    """
    変数が「空」かどうかのチェック
        変数型ごとに判断基準・判断方法が異なる
        ため、必要に応じて実装する
        仮に、一致するものがない場合は、その変数
        の真偽値表現 (bool() でキャストした結果) を
        返却する
    
    Parameters
    ----------
    var : any
        チェックする変数

    Returns
    -------
    bool
        「空」かどうか
    """
    
    # TODO: imageMb 型のチェック
    
    if isinstance( var, Number ):
        # 数値型の場合、数値にかかわらず False
        return False
    
    elif isinstance( var, str ):
        # 空文字の場合 True
        return var == ""
    
    elif isinstance( var, (list, tuple, dict, set) ):
        # 要素が0の場合 True
        return len( var ) == 0
    
    elif isinstance( var, np.ndarray ):
        # 大きさが0の場合 True
        return var.size == 0
    
    else:
        # 実装依存
        return not bool( var )


def EMPTY_VALUE_ASSERT( var ):
    """
    変数の空値アサーションテンプレート
    
    Parameters
    ----------
    var : any
        空値チェックを適応したい変数
    
    Returns
    -------
    """
    message = "argument '{var_name}' must not be empty value, ({var_name} = {value})".format(
        var_name=_get_var_name( var ),
        value=var
    )
    
    if __debug__:
        if _IS_EMPTY( var ):
            raise AssertionError( message )


# 型アサーションのテンプレート
def TYPE_ASSERT( var, types, allow_empty=False ):
    """
    
    変数の型アサーションテンプレート
    
    Parameters
    ----------
    var : any
        型アサーションを適応したい変数
    types : type or tuple of type
        変数 `var` について許容される型の一覧

    Returns
    -------

    """
    
    def _unpack_types( _types ):
        _unpacked = list()
        
        for _type in _types:
            if isinstance( _type, _Iterable ):
                _unpacked += _type
            else:
                if _type is None:
                    _type = type( None )
                
                _unpacked.append( _type )
        
        return tuple( _unpacked )
    
    
    if isinstance( types, _Iterable ):
        types = _unpack_types( types )
        
        message = "argument '{var_name}' must be {type_str}, not {var_class}".format(
            var_name=_get_var_name( var ),
            type_str=' or '.join( [_get_qualified_class_name( t ) for t in types] ),
            var_class=_get_qualified_class_name( type( var ) )
        )
    else:
        message = "argument '{var_name}' must be {type_str}, not {var_class}".format(
            var_name=_get_var_name( var ),
            type_str=_get_qualified_class_name( types ),
            var_class=_get_qualified_class_name( type( var ) )
        )
    
    if __debug__:
        if not isinstance( var, types ):
            raise AssertionError( message )
        if not allow_empty and var is not None:
            EMPTY_VALUE_ASSERT( var )


# numpy.ndarray: 次元数(ndim)についてのアサーション
def NDIM_ASSERT( ndarray, ndim ):
    """
    
    numpy.ndarray: 次元数(`ndim`)についてのアサーション
    
    Parameters
    ----------
    ndarray : numpy.ndarray
        アサーションを適応したい変数
    ndim : int
        次元数

    Returns
    -------

    """
    TYPE_ASSERT( ndarray, np.ndarray )
    
    message = "argument '{var_name}' must be 'numpy.ndarray' with {n}-dimension".format(
        var_name=_get_var_name( ndarray ),
        n=str( ndim )
    )
    
    if __debug__:
        if not ndarray.ndim == ndim:
            raise AssertionError( message )


# numpy.ndarray: ndarray の属性についてのアサーション
def NDARRAY_ASSERT( ndarray, **kwargs ):
    """
    
    numpy.ndarray: 次元数(`ndim`)についてのアサーション
    
    Parameters
    ----------
    ndarray : numpy.ndarray
        アサーションを適応したい変数
    **kwargs
        属性(大きさ、データタイプ)

    Returns
    -------

    """
    TYPE_ASSERT( ndarray, np.ndarray )
    
    message = "argument '{var_name}' must be 'numpy.ndarray(params)'".format(
        var_name=_get_var_name( ndarray ),
        params=', '.join( ["{}={}".format( k, v ) for k, v in kwargs.items()] )
    )
    
    if __debug__:
        for k, v in kwargs.items():
            if ndarray.__getattribute__( k ) != v:
                raise AssertionError( message )


def SAME_NDIM_ASSERT( ndarray_1, ndarray_2 ):
    """

    numpy.ndarray: 次元数の一致についてのアサーション

    Parameters
    ----------
    ndarray_1, ndarray_2 : numpy.ndarray
        `ndim` が比較される変数
    
    Returns
    -------

    """
    
    TYPE_ASSERT( ndarray_1, np.ndarray )
    TYPE_ASSERT( ndarray_2, np.ndarray )
    
    message = "argument '{var_name_1}' and '{var_name_2}' must be dimension ({var_name_1}.ndim = {n_1}, {var_name_2}.ndim = {n_2})".format(
        var_name_1=_get_var_name( ndarray_1 ),
        n_1=str( ndarray_1.ndim ),
        var_name_2=_get_var_name( ndarray_2 ),
        n_2=str( ndarray_2.ndim )
    )
    
    if __debug__:
        if not ndarray.ndim == ndim:
            raise AssertionError( message )


def SAME_SHAPE_ASSERT( ndarray_1, ndarray_2, ignore_ndim=False ):
    """

    numpy.ndarray: 配列サイズの一致についてのアサーション

    Parameters
    ----------
    ndarray_1, ndarray_2 : numpy.ndarray
        `shape` が比較される変数
    
    Returns
    -------

    """
    
    TYPE_ASSERT( ndarray_1, np.ndarray )
    TYPE_ASSERT( ndarray_2, np.ndarray )
    
    if ignore_ndim:
        min_ndim = min( ndarray_1.ndim, ndarray_2.ndim )
        shape_1 = ndarray_1.shape[:min_ndim]
        shape_2 = ndarray_2.shape[:min_ndim]
    else:
        shape_1, shape_2 = ndarray_1.shape, ndarray_2.shape
    
    message = "argument '{var_name_1}' and '{var_name_2}' must be same shape ({var_name_1}.shape = {shape_1}, {var_name_2}.shape = {shape_2})".format(
        var_name_1=_get_var_name( ndarray_1 ),
        shape_1=shape_1,
        var_name_2=_get_var_name( ndarray_2 ),
        shape_2=shape_2,
    )
    
    if __debug__:
        if not shape_1 == shape_2:
            raise AssertionError( message )


def dec_debug( func ):
    """
    Decorator: 関数デバッグ用
        - 関数呼び出し時と処理終了時に
          標準エラー出力に出力を行う
          
    Parameters
    ----------
    func : callable object
        デコレータを適用したい関数

    Returns
    -------
    callable object
        デコレータ処理を行う関数
    """
    
    def wrapper( *args, **kwargs ):
        eprint( "{func_name}: Calculating ... ".format(
            func_name=func.__name__
        ) )
        ret_val = func( *args, **kwargs )
        eprint( "{func_name}: finished !".format(
            func_name=func.__name__
        ) )
        
        return ret_val
    
    return wrapper


def compute_by_window( imgs, func, window_size=16, step=2, dst_dtype=np.float32 ):
    """
    画像を一部を切り取り、func 関数で行った計算結果を
    返却する

    Parameters
    ----------
    imgs : numpy.ndarray or tuple of numpy.ndarray
        入力画像
        tuple で複数画像を与える場合、各画像に対して
        同じ領域を切り取り、処理を行うため、各画像の
        縦、横サイズは一致している必要がある
    func : callable object
        切り取った画像の一部に対して何らかの計算を行う
        関数。引数として画像の一部が渡される。
    window_size : int or tuple of int
        画像を切り取るサイズ。
        int を指定した場合は、縦横同じサイズで切り取る。
        tuple(int, int) を指定した場合は、縦横で異なったサイズ
        で切り取り、指定する順序は ndarray の次元に対応する
    step : int or tuple of int
        切り取り間隔
        int を指定した場合は、縦横同じ間隔を開けて処理をする
        tuple(int, int) を指定した場合は、縦横で異なった間隔
        を開けて処理を行い、指定する順序は ndarray の次元に
        対応する

    Returns
    -------
    numpy.ndarray
        各切り取り画像に対する処理結果の行列
    """
    TYPE_ASSERT( imgs, [np.ndarray, tuple] )
    
    if isinstance( imgs, tuple ):
        for img in imgs:
            TYPE_ASSERT( img, np.ndarray )
        for i in range( len( imgs ) - 1 ):
            SAME_SHAPE_ASSERT( imgs[i], imgs[i + 1] )
        n_imgs = len( imgs )
        height, width = imgs[0].shape[:2]
    else:
        n_imgs = 1
        height, width = imgs.shape[:2]
    
    assert callable( func ) and func.__code__.co_argcount >= n_imgs, \
        "argument 'func' must be callable object which has {0} argument at least. \n".format( n_imgs ) + \
        "  ( num of argumets of 'func' depends on argument 'imgs')"
    
    TYPE_ASSERT( window_size, [int, tuple] )
    TYPE_ASSERT( step, [int, tuple] )
    
    if isinstance( step, int ):
        s_i, s_j = [step] * 2
    else:
        s_i, s_j = step
    
    if isinstance( window_size, int ):
        w_i, w_j = [window_size] * 2
    else:
        w_i, w_j = window_size
    
    results = np.ndarray(
        (
            ceil( height / s_i ),
            ceil( width / s_j )
        ),
        dtype=dst_dtype
    )
    
    for ii, i in enumerate( range( 0, height, s_i ) ):
        
        for jj, j in enumerate( range( 0, width, s_j ) ):
            
            eprint( "\rWindow calculating ... {1:{0}d} / {2:{0}d}".format(
                len( str( results.size ) ),
                (jj + 1) + ii * results.shape[1],
                results.size
            ), end="" )
            
            rois = [
                img[i: i + w_i, j: j + w_j]
                for img in imgs
            ]
            
            results[ii][jj] = func( *rois )
    
    return results
