"""
Functions to help with task-based parallel execution.
"""
from __future__ import (division, unicode_literals, absolute_import,
                        print_function)

import multiprocessing as mp


def run_func(func, args, parallel=False):
    """
    Run a function with a list of argument sets, either in serial or parallel.

    Parameters
    ----------
    func: function
        The function to run
    args:
        A list of objects, where each entry `a` can be used like `func(a)`
    parallel: bool
        Whether to run the tasks in parallel.
    """
    if parallel:
        mp.Pool(mp.cpu_count() - 1).map(func, args)
    else:
        for arg in args:
            func(arg)
