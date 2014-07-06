'''
Distance finding functions
inspired by scipy.spatial.distance
defined for periodic systems.
'''

import numpy as np


def pdist_periodic(r, L):
    '''
    Return the square-euclidean distance between all combinations of
    a set of points,
    in a periodic space of length L.
    '''
    d = csep_periodic(r, r, L)
    d[np.identity(len(r), dtype=np.bool)] = np.inf
    d_sq = np.sum(np.square(d), axis=-1)
    return d_sq


def csep_periodic(ra, rb, L):
    '''
    Find all separation vectors between ra and rb,
    in a periodic system of length L.
    '''
    seps = ra[:, np.newaxis, :] - rb[np.newaxis, :, :]
    seps[seps > L / 2.0] -= L
    seps[seps < -L / 2.0] += L
    return seps


def cdist_periodic(ra, rb, L):
    '''
    Find all square-euclidean distances between ra and rb,
    in a periodic system of length L.
    '''
    return np.sum(np.square(csep_periodic(ra, rb, L)), axis=-1)


def csep_periodic_close(ra, rb, L):
    '''
    Find closest separation vectors of ra to rb,
    in a periodic system of length L.
    '''
    seps = csep_periodic(ra, rb, L)
    seps_sq = np.sum(np.square(seps), axis=-1)

    i_close = np.argmin(seps_sq, axis=-1)

    i_all = list(range(len(seps)))
    sep = seps[i_all, i_close]
    sep_sq = seps_sq[i_all, i_close]
    return sep, sep_sq
