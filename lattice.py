import numpy as np


def extend_array(a, n):
    '''
    Increase the resolution of an array by duplicating its values to fill
     a larger array.

     Parameters
     ----------
     a: array, shape (a1, a2, a3, ...)
     n: integer
        Factor by which to expand the array.

    Returns
    -------
    ae: array, shape (n * a1, n * a2, n * a3, ...)
    '''
    a_new = a.copy()
    for d in range(a.ndim):
        a_new = np.repeat(a_new, n, axis=d)
    return a_new


def field_subset(f, inds, rank=0):
    '''
    Return the value of a field at a subset of points.

    Parameters
    ----------
    f: array, shape (a1, a2, ..., ad, r1, r2, ..., rrank)
        Rank-r field in d dimensions
    inds: integer array, shape (n, d)
        Index vectors
    rank: integer
        The rank of the field (0: scalar field, 1: vector field and so on).

    Returns
    -------
    f_sub: array, shape (n, rank)
        The subset of field values.
    '''
    f_dim_space = f.ndim - rank
    if inds.ndim > 2:
        raise Exception('Too many dimensions in indices array')
    if inds.ndim == 1:
        if f_dim_space == 1:
            return f[inds]
        else:
            raise Exception('Indices array is 1d but field is not')
    if inds.shape[1] != f_dim_space:
        raise Exception('Indices and field dimensions do not match')
    # It's magic, don't touch it!
    return f[tuple([inds[:, i] for i in range(inds.shape[1])])]


def pad_to_3d(a):
    '''
    Convert and return 1- or 2-dimensional cartesian vectors into a
    3-dimensional representation, with additional dimensional coordinates
    assumed to be zero.

    Parameters
    ----------
    a: array, shape (n, d), d < 3

    Returns
    -------
    ap: array, shape (n, 3)
    '''
    a_pad = np.zeros([len(a), 3], dtype=a.dtype)
    a_pad[:, :a.shape[-1]] = a
    return a_pad


def pad_length(x, d):
    '''
    Convert a scalar value to a vector form appropriate to a dimensional space,
    if needed. If the input is a vector, leave it untouched.
    Useful when a function expects an array specifying lengths along each axis,
    but wants to also accept a scalar value in case the length is the same in
    all directions.

    Parameters
    ----------
    x: float or array-like
        The input parameter that may need padding.
    d: int
        The dimensional space to make `x` appropriate for.

    Returns
    -------
    x_pad: array-like, shape (d,)
        The padded parameter.
    '''
    try:
        x[0]
    except TypeError:
        x = d * [x]
    return np.array(x)


def wrap_inc(M, i):
    return i + 1 if i < M - 1 else 0


def wrap_dec(M, i):
    return i - 1 if i > 0 else M - 1


def r_to_i(r, L, dx):
    '''
    Return closest indices on a square lattice of vectors in continuous space.

    Parameters
    ----------
    r: float array, shape (a1, a2, ..., d)
        Cartesian vectors, with last axis indexing the dimension.
        It's assumed that all components of the vector lie within
        plus or minus L / 2.
    L: float
        Length of the lattice, assumed to be centred on the origin.
    dx: float
        Spatial lattice spacing.
        This means the number of lattice points is (L / dx).

    Returns
    -------
    inds: integer array, shape of r
    '''
    return np.asarray((r + L / 2.0) / dx, dtype=np.int)


def i_to_r(i, L, dx):
    '''
    Return coordinates of lattice indices in continuous space.

    Parameters
    ----------
    i: integer array, shape (a1, a2, ..., d)
        Integer indices, with last axis indexing the dimension.
        It's assumed that all components of the vector
        lie within plus or minus (L / dx) / 2.
    L: float
        Length of the lattice, assumed to be centred on the origin.
    dx: float
        Spatial lattice spacing.
        This means the number of lattice points is (L / dx).

    Returns
    -------
    r: float array, shape of i
        Coordinate vectors of the lattice points specified by the indices.

    '''
    return -L / 2.0 + (i + 0.5) * dx
