'''
Distance finding functions inspired by scipy.spatial.distance.
'''

from __future__ import print_function, division
import numpy as np


def csep(ra, rb):
    '''
    Returns separation vectors between each pair of the two sets of points.

    Parameters
    ----------
    ra, rb: float array-like, shape (n, d) and (m, d) in d dimensions.
        Two sets of points.

    Returns
    -------
    csep: float array-like, shape (n, m, d)
        csep[i, j] is the separation vector from point j to point i.
        Note the un-intuitive vector direction.
    '''
    return ra[:, np.newaxis, :] - rb[np.newaxis, :, :]


def csep_close(ra, rb):
    '''
    Returns the closest separation vector between each point in one set,
    and every point in a second set.

    Parameters
    ----------
    ra, rb: float array-like, shape (n, d) and (m, d) in d dimensions.
        Two sets of points. `ra` is the set of points from which the closest
        separation vectors to points `rb` are calculated.

    Returns
    -------
    csep_close: float array-like, shape (n, m, d)
        csep[i] is the closest separation vector from point ra[j]
        to any point rb[i].
        Note the un-intuitive vector direction.
    '''
    seps = csep(ra, rb)
    seps_sq = np.sum(np.square(seps), axis=-1)

    i_close = np.argmin(seps_sq, axis=-1)

    i_all = list(range(len(seps)))
    sep = seps[i_all, i_close]
    sep_sq = seps_sq[i_all, i_close]
    return sep, sep_sq


def csep_periodic(ra, rb, L):
    '''
    Returns separation vectors between each pair of the two sets of points.

    Parameters
    ----------
    ra, rb: float array-like, shape (n, d) and (m, d) in d dimensions.
        Two sets of points.
    L: float
        System size.

    Returns
    -------
    csep: float array-like, shape (n, m, d)
        csep[i, j] is the separation vector from point j to point i.
        Note the un-intuitive vector direction.
    '''
    try:
        L[0]
    except (TypeError, IndexError):
        L = np.ones([ra.shape[1]]) * L
    seps = ra[:, np.newaxis, :] - rb[np.newaxis, :, :]
    for i_dim in range(ra.shape[1]):
        seps_dim = seps[:, :, i_dim]
        seps_dim[seps_dim > L[i_dim] / 2.0] -= L[i_dim]
        seps_dim[seps_dim < -L[i_dim] / 2.0] += L[i_dim]
    return seps


def csep_periodic_close(ra, rb, L):
    '''
    Returns the closest separation vector between each point in one set,
    and every point in a second set, in periodic space.

    Parameters
    ----------
    ra, rb: float array-like, shape (n, d) and (m, d) in d dimensions.
        Two sets of points. `ra` is the set of points from which the closest
        separation vectors to points `rb` are calculated.
    L: float
        System size.

    Returns
    -------
    csep_close: float array-like, shape (n, m, d)
        csep[i] is the closest separation vector from point ra[j]
        to any point rb[i].
        Note the un-intuitive vector direction.
    '''
    seps = csep_periodic(ra, rb, L)
    seps_sq = np.sum(np.square(seps), axis=-1)

    i_close = np.argmin(seps_sq, axis=-1)

    i_all = list(range(len(seps)))
    sep = seps[i_all, i_close]
    sep_sq = seps_sq[i_all, i_close]
    return sep, sep_sq


def cdist_sq_periodic(ra, rb, L):
    '''
    Returns the squared distance between each point in on set,
    and every point in a second set, in periodic space.

    Parameters
    ----------
    ra, rb: float array-like, shape (n, d) and (m, d) in d dimensions.
        Two sets of points.
    L: float
        System size.

    Returns
    -------
    cdist_sq: float array-like, shape (n, m, d)
        cdist_sq[i, j] is the squared distance between point j and point i.
    '''
    return np.sum(np.square(csep_periodic(ra, rb, L)), axis=-1)


def pdist_sq_periodic(r, L):
    '''
    Returns the squared distance between all combinations of
    a set of points, in periodic space.

    Parameters
    ----------
    r: shape (n, d) for n points in d dimensions.
        Set of points
    L: float
        System size.

    Returns
    -------
    d_sq: float array, shape (n, n, d)
        Squared distances
    '''
    d = csep_periodic(r, r, L)
    d[np.identity(len(r), dtype=np.bool)] = np.inf
    d_sq = np.sum(np.square(d), axis=-1)
    return d_sq
