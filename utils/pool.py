#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2020/01/17
# This is a part of EQDmgAnalyzr

# /usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: popunbom <fantom0779@gmail.com>
# Created At: 2019/12/06
import atexit
from os.path import exists
from os import cpu_count, remove
from multiprocessing import Pool, current_process
from textwrap import dedent

from utils.common import eprint


class CustomPool:
    """
    プロセスワーカー ID を正しく設定するために
    カスタマイズしたプロセスプール
    """
    
    PATH_N_PROCESS = "./n_process"
    
    
    def __init__(self, worker_name=""):
        """
        コンストラクタ
        Parameters
        ----------
        worker_name : str, default ""
            プロセスワーカーの識別子
            設定しない場合、デフォルト値が設定される
        """
        
        self.n_process = None
        self.worker_name = worker_name
        
        if exists(self.PATH_N_PROCESS):
            with open(self.PATH_N_PROCESS) as f:
                # eprint("File Loaded --", self.PATH_N_PROCESS)
                self.n_spawned = int(f.read().strip("\n"))
        else:
            self.n_spawned = 0
        
        atexit.register(self._cleanup)
        
        # eprint(dedent(f"""
        # ---------- CustomPool.__init__() ----------
        # self.n_process: {self.n_process}
        # self.worker_name: {self.worker_name}
        # self.n_spawned: {self.n_spawned}
        # -------------------------------------------
        # """).rstrip("\n"))
    
    
    def _cleanup(self):
        eprint("CustomPool: cleanup")
        if exists(self.PATH_N_PROCESS):
            # eprint(
            #     "File will be removed -- {path}".format(
            #         path=self.PATH_N_PROCESS
            #     )
            # )
            remove(self.PATH_N_PROCESS)
    
    
    def _initializer(self, n_process, _initializer, _initargs):
        """
        ワーカー初期化関数 (イニシャライザ)
        
        Parameters
        ----------
        n_process : int
            生成されるワーカープロセスの数
        _initializer : callable object
            初期化処理
            この関数の最後に実行される
        _initargs : list or tuple
            引数の関数 _initializer に
            渡される関数リスト
        """
        
        p = current_process()
        
        p._identity = tuple([
            p._identity[0] - self.n_spawned,
            *p._identity[1:]
        ])
        
        # eprint(dedent(f"""
        # ---------- CustomPool._initializer() ----------
        # current_process: {p}
        # current_process._identity: {p._identity}
        # -----------------------------------------------
        # """).rstrip("\n"))
        
        _initializer(*_initargs)
    
    
    def update(self):
        """
        更新関数
        
        - 使用されたプロセスプールで生成された
          ワーカープロセスの数を累積することで、
          ワーカーID の設定を正しく行う
        Returns
        -------

        """
        if self.n_process is None:
            raise ValueError(
                "'update' must be run after generating 'Pool'"
            )
        
        self.n_spawned += self.n_process
        
        with open(self.PATH_N_PROCESS, "w") as f:
            f.write(
                str(self.n_spawned)
            )
        
        self.n_process = None
        
        # eprint(dedent(f"""
        # ---------- CustomPool.update() ----------
        # self.n_process: {self.n_process}
        # self.n_spawned: {self.n_spawned}
        # -----------------------------------------------
        # """).rstrip("\n"))
    
    
    def Pool(self, n_process=None, **kwargs):
        """
        プロセスプールの生成
        
        Parameters
        ----------
        n_process : int, default None
            生成されるワーカープロセスの数
            指定されなかった場合、os.cpu_count() の
            値を使用する
        kwargs : dict
            multiprocess.Pool 生成時に使用する引数

        Returns
        -------
        
        multiprocess.Pool
            プロセスプール

        """
        _initializer = None
        _initargs = None
        
        if "initializer" in kwargs:
            _initializer = kwargs["initializer"]
        if "initargs" in kwargs:
            _initargs = kwargs["initargs"]
        
        if n_process is None:
            n_process = cpu_count()
        
        kwargs["initializer"] = self._initializer
        kwargs["initargs"] = tuple([
            n_process,
            _initializer,
            _initargs
        ])
        
        self.n_process = n_process
        return Pool(n_process, **kwargs)
