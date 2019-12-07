#/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: popunbom <fantom0779@gmail.com>
# Created At: 2019/12/06

from time import sleep
from tqdm import trange, tqdm
from multiprocessing import Pool, freeze_support, current_process
import re

def progresser(n):
    interval = 0.001 / (n + 2)
    total = 5000
    worker_id = int(re.match(r"(.*)-([0-9]+)$", current_process().name).group(2))
    # tqdm.write("Worker: " + str(current_process()) + ", PID: " + str(worker_id))
    text = "#{}, est. {:<04.2}s".format(n, interval * total)
    for _ in trange(total, desc=text, position=worker_id, leave=False):
    # for _ in range(total):
        sleep(interval)

if __name__ == '__main__':
    
    # pool = Pool(2)
    # for _ in tqdm.tqdm(pool.imap_unordered(progresser, range(100)), total=100):
    #     pass
    #
    # pool.close()
    # pool.join()
    # pbar.close()

    with Pool(processes=4, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as p:
        for result in tqdm(p.imap_unordered(progresser, range(10)), total=10):
            pass
