"""Functions related to statistics and uncertainties"""
from __future__ import (division, unicode_literals, absolute_import,
                        print_function)

from scipy.stats import chi2_contingency
import numpy as np


def sample_var_var(std, n):
    """
    The variance of the sample variance of a distribution.

    Assumes the samples are normally distributed.

    From: //math.stackexchange.com/q/72975

    `std`: Distribution's standard deviation
    `n`: Number of samples
    """
    return 2.0 * std ** 4 / (n - 1.0)


def g_test(observed_frequencies):
    return chi2_contingency(observed_frequencies, correction=True,
                            lambda_='log-likelihood')


def p_subset_different(nr_A_sub, nr_A_all, nr_B_sub, nr_B_all):
    contin = np.array([
                       [nr_A_sub, nr_A_all - nr_A_sub],
                       [nr_B_sub, nr_B_all - nr_B_sub],
                      ])
    try:
        test_stat, p, dof, expected = g_test(contin)
    except ValueError:
        p = np.nan
    return p


def p_subset_different_row(row, sub_A_col, sub_B_col, all_A_col, all_B_col):
    return p_subset_different(row[sub_A_col], row[all_A_col],
                              row[sub_B_col], row[all_B_col])


def weighted_covariance(x, y, w):
    """Weighted Covariance"""
    return np.sum(w *
                  (x - np.average(x, weights=w)) *
                  (y - np.average(y, weights=w))) / np.sum(w)


def weighted_correlation(x, y, w):
    """Weighted Correlation"""
    return (weighted_covariance(x, y, w) /
            np.sqrt(weighted_covariance(x, x, w) *
                    weighted_covariance(y, y, w)))


def normalize(v):
    return (v - v.mean()) / v.std()


def bootstrap_statistic(v, stat_func, n_samples=100, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()
    stats = []
    for i in range(n_samples):
        v_ = v.sample(frac=1.0, replace=True, random_state=random_state)
        stats.append(stat_func(v_))
    return stats


def bootstrap_percentile_err(v, percentile, stat_func, offsets=False,
                             *args, **kwargs):
    if len(v) == 0:
        return np.nan
    stats = bootstrap_statistic(v, stat_func, *args, **kwargs)
    values = np.percentile(stats, percentile)
    if offsets:
        values -= stat_func(v)
    return values
