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


def eprint( *args, end="\n" ):
    msg = ' '.join( [str( e ) for e in args] ) + end
    sys.stderr.write( msg )
    sys.stderr.flush()


class CannotFindVariable( Exception ):
    
    
    def __init__( self, *args: object ) -> None:
        super().__init__( *args )


class InvalidImageOrFile( Exception ):
    
    
    def __init__( self, *args: object ) -> None:
        super().__init__( *args )


# 変数から変数名の取得
def _get_var_name( var ):
    for fi in reversed( inspect.stack() ):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len( names ) > 0:
            return "'" + names[0] + "'"
    else:
        raise CannotFindVariable( "Cannot find variable name" )


# オブジェクトの完全クラス名の取得
def _get_qualfied_class_name( _type ):
    is_not_builtins = hasattr( _type, "__module__" ) and _type.__module__ != "builtins"
    
    if is_not_builtins:
        return "'{module_name}.{class_name}'".format(
            module_name=_type.__module__,
            class_name=_type.__name__
        )
    else:
        return _type.__name__


# 変数が「空」かどうかのチェック
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
    message = "argument '{var_name}' must not be empty value, ({var_name} = {value}".format(
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
                _unpacked.append( _type )
        
        return tuple(_unpacked)
        
    
    if isinstance( types, _Iterable ):
        types = _unpack_types( types )
        
        message = "argument '{var_name}' must be {type_str}, not {var_class}".format(
            var_name=_get_var_name( var ),
            type_str=' or '.join( [_get_qualfied_class_name( t ) for t in types] ),
            var_class=_get_qualfied_class_name( type( var ) )
        )
    else:
        message = "argument '{var_name}' must be {type_str}, not {var_class}".format(
            var_name=_get_var_name( var ),
            type_str=_get_qualfied_class_name( types ),
            var_class=_get_qualfied_class_name( type( var ) )
        )
    
    if __debug__:
        if not isinstance( var, types ):
            raise AssertionError( message )
        if not allow_empty:
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
        params=', '.join( [ "{}={}".format(k, v) for k, v in kwargs.items() ] )
    )
    
    if __debug__:
        for k, v in kwargs.items():
            if ndarray.__getattribute__(k) != v:
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
        shape_1, shape2 = ndarray_1.shape, ndarray_2.shape
    
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
