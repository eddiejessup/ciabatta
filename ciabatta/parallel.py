import multiprocessing as mp


def run_func(func, args, parallel=False):
    if parallel:
        mp.Pool(mp.cpu_count() - 1).map(func, args)
    else:
        for arg in args:
            func(arg)
