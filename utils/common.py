# /usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: popunbom <fantom0779@gmail.com>
# Created At: 2019/08/29
import functools
import sys
import traceback
from multiprocessing import current_process

from tqdm import tqdm

from utils.assertion import TYPE_ASSERT


def eprint( *args, end="\n", debug_flag=None, use_tqdm=False ):
    """
    print 関数の標準エラー出力版
    Parameters
    ----------
    args : list
        出力したい変数
    end : str, default "\n"
        行末に付加される文字列
    debug_flag : None or bool
        デバッグモードかどうか
        - None または True の場合に出力される
    use_tqdm : bool
        tqdm を使用するかどうか
        - tqdm のループ内で使用する場合は True にする必要がある

    Returns
    -------
    """
    TYPE_ASSERT( debug_flag, [None, bool] )
    
    if debug_flag is None or debug_flag:
        msg = ' '.join( [str( e ) for e in args] )
        
        if use_tqdm:
            tqdm.write( msg, end=end )
        else:
            sys.stderr.write( msg + end )
            sys.stderr.flush()


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


def check_module_avaliable( lib_name ):
    """
    モジュールが利用可能かチェックする
    Parameters
    ----------
    lib_name : str
        モジュール名

    Returns
    -------
    bool
        モジュールが利用可能かどうか
    """
    
    from importlib.util import find_spec
    
    return find_spec( lib_name ) is not None


def worker_exception_raisable(func):
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            # print 'Exception in ' + func.__name__
            eprint(
                "[{worker_name}]: Exception raised !".format(
                    worker_name=current_process().name
                )
            )
            traceback.print_exc()
    
    return wrapper
