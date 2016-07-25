"""
Miscellaneous random number-related functions.
"""
from __future__ import (division, unicode_literals, absolute_import,
                        print_function)

import numpy as np


def randbool(n, rng=None):
    """Return a number of random samples from the set {-1, +1}.

    Parameters
    ----------
    n: int
        Number of samples
    rng: np.random.RandomState
        Random number generator. If not provided, use numpy's default.
    """
    if rng is None:
        rng = np.random
    return rng.randint(2, size=n) * 2 - 1
