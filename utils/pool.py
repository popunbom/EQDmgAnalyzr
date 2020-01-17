#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2020/01/17
# This is a part of EQDmgAnalyzr

# /usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: popunbom <fantom0779@gmail.com>
# Created At: 2019/12/06

from time import sleep
from tqdm import trange, tqdm
from multiprocessing import Pool, current_process
import re


N_POOL_SPAWNED_WORKER = 0


class CustomPool:
    
    
    def __init__(self, worker_name=""):
        self.n_process = None
        self.worker_name = worker_name
    
    
    def _initializer(self, n_process, _initializer, _initargs):
        global N_POOL_SPAWNED_WORKER
        
        print("N_POOL_SPAWNED_WORKER:", N_POOL_SPAWNED_WORKER)
        
        p = current_process()
        
        print("Process:", p)
        print("_identity (before):", p._identity)
        
        p._identity = tuple([
            p._identity[0] - N_POOL_SPAWNED_WORKER,
            *p._identity[1:]
        ])
        
        print("_identity (after) :", p._identity)
        
        _initializer(*_initargs)
    
    
    def update(self):
        global N_POOL_SPAWNED_WORKER
        
        if self.n_process is None:
            raise ValueError(
                "'update' must be run after generating 'Pool'"
            )
        
        N_POOL_SPAWNED_WORKER += self.n_process
        
        self.n_process = None
    
    
    def Pool(self, n_process, **kwargs):
        _initializer = None
        _initargs = None
        
        if "initializer" in kwargs:
            _initializer = kwargs["initializer"]
        if "initargs" in kwargs:
            _initargs = kwargs["initargs"]
        
        kwargs["initializer"] = self._initializer
        kwargs["initargs"] = tuple([
            n_process,
            _initializer,
            _initargs
        ])
        
        self.n_process = n_process
        
        return Pool(n_process, **kwargs)
