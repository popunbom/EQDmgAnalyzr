# /usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: popunbom <fantom0779@gmail.com>
# Created At: 2019/08/29

import sys
import inspect

from .exception import CannotFindVariable


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


def get_var_name( var ):
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


def get_qualified_class_name( _type ):
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


def n_args( func ):
    return func.__code__.co_argcount
