'''
Distance finding functions
inspired by scipy.spatial.distance.
'''

import numpy as np


def csep(ra, rb):
    '''
    Find all separation vectors between ra and rb.
    '''
    seps = ra[:, np.newaxis, :] - rb[np.newaxis, :, :]
    return seps


def csep_close(ra, rb):
    '''
    Find closest separation vectors of ra to rb.
    '''
    seps = csep(ra, rb)
    seps_sq = np.sum(np.square(seps), axis=-1)

    i_close = np.argmin(seps_sq, axis=-1)

    i_all = list(range(len(seps)))
    sep = seps[i_all, i_close]
    sep_sq = seps_sq[i_all, i_close]
    return sep, sep_sq
