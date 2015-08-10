"""
Miscellaneous random number-related functions.
"""
from __future__ import print_function, division
import numpy as np


def randbool(n, rng=None):
    if rng is None:
        rng = np.random
    return rng.randint(2, size=n) * 2 - 1
