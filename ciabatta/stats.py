"""Functions related to statistics and uncertainties"""
from __future__ import (division, unicode_literals, absolute_import,
                        print_function)


def sample_var_var(std, n):
    """
    The variance of the sample variance of a distribution.

    Assumes the samples are normally distributed.

    From: //math.stackexchange.com/q/72975

    `std`: Distribution's standard deviation
    `n`: Number of samples
    """
    return 2.0 * std ** 4 / (n - 1.0)
