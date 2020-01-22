#/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: popunbom <fantom0779@gmail.com>
# Created At: 2019/12/06

from time import sleep
from tqdm import trange, tqdm
from multiprocessing import Pool, current_process
import re

from utils.pool import CustomPool


def progresser_1(n):
    interval = 0.001 / (n + 2)
    worker_id = current_process()._identity[0]
    text = f"Worker #{worker_id} "
    
    total = 5000
    for _ in trange(total, desc=text, position=worker_id, leave=False):
        sleep(interval)


def progresser_2(n):
    worker_id = current_process()._identity[0]
    text = f"Worker #{worker_id} "
    
    total = 500000
    s = 0
    for _ in trange(total, desc=text, position=worker_id, leave=False):
        s += s * (s % 3)


def proc_1():
    cp = CustomPool()
    
    with cp.Pool(n_process=4, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as p:
        for result in tqdm(p.imap_unordered(progresser_1, range(10)), total=10):
            pass
    
    cp.update()


def proc_2():
    cp = CustomPool()
    
    with cp.Pool(n_process=4, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as p:
        for result in tqdm(p.imap_unordered(progresser_2, range(10)), total=10):
            pass
    
    cp.update()


if __name__ == '__main__':
    proc_1()
    proc_2()
