#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2019-11-24
# This is a part of EQDmgAnalyzr
import inspect

from utils.exception import CannotFindVariable


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
