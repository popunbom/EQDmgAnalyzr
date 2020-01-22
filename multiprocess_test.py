import re
from multiprocessing import Pool, current_process
from random import randint, seed

from tqdm import trange, tqdm
from uuid import uuid4


def unique_str(n=8):
    return str(uuid4()).replace("-", "")[:n]


def func():
    desc = ""
    # worker_id = int(re.match(r"(.*)-([0-9]+)$", current_process().name).group(2))
    # desc = f"Worker #{worker_id:3d}"
    
    # tqdm.write(str(current_process().pid))
    print(str(current_process().pid))
    seed()
    s = 0
    # for i in trange(0, 250000, desc=desc, leave=False):
    for i in range(0, 250000):
        s += i % (randint(1, i+1))
    
    return s


def mp_loop():
    L = list(range(100))
    
    # progress_bar = tqdm(total=len(L))
    
    
    def _update_progressbar(arg):
        pass
        # progress_bar.update()
        
    def _initialize(proc_name):
        tqdm.set_lock(tqdm.get_lock())
        current_process().name = proc_name
    
    print("pid:", current_process().pid)
    # pool = Pool(processes=6, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
    pool = Pool(processes=6, initializer=_initialize, initargs=(unique_str(),))
    
    results = list()
    for _ in L:
        results.append(
            pool.apply_async(
                func,
                callback=_update_progressbar
            )
        )
    pool.close()
    pool.join()
    
    results = [result.get() for result in results]
    
    pool.terminate()
    
    print(results)


if __name__ == '__main__':
    mp_loop()
    mp_loop()
