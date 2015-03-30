from __future__ import print_function, division
import numpy as np


def vector_mag_sq(v):
    '''
    Returns the squared magnitude of vectors.

    Parameters
    ----------
    v: array, shape (a1, a2, ..., d)
        Cartesian vectors, with last axis indexing the dimension.

    Returns
    -------
    mag: array, shape (a1, a2, ...)
        Vector squared magnitudes
    '''
    return np.square(v).sum(axis=-1)


def vector_mag(v):
    '''
    Returns the magnitude of vectors.

    Parameters
    ----------
    v: array, shape (a1, a2, ..., d)
        Cartesian vectors, with last axis indexing the dimension.

    Returns
    -------
    mag: array, shape (a1, a2, ...)
        Vector magnitudes
    '''
    return np.sqrt(vector_mag_sq(v))


def vector_unit_nonull(v):
    '''
    Returns unit vectors of input vectors.
    Any null vectors raise an Exception

    Parameters
    ----------
    v: array, shape (a1, a2, ..., d)
        Cartesian vectors, with last axis indexing the dimension.

    Returns
    -------
    v_new: array, shape of v
    '''
    if v.size == 0:
        return v
    return v / vector_mag(v)[..., np.newaxis]


def vector_unit_nullnull(v):
    '''
    Returns unit vectors of input vectors.
    Any null vectors remain null vectors.

    Parameters
    ----------
    v: array, shape (a1, a2, ..., d)
        Cartesian vectors, with last axis indexing the dimension.

    Returns
    -------
    v_new: array, shape of v
    '''
    if v.size == 0:
        return v
    mag = vector_mag(v)
    v_new = v.copy()
    v_new[mag > 0.0] /= mag[mag > 0.0][..., np.newaxis]
    return v_new


def vector_unit_nullrand(v):
    '''
    Returns unit vectors of input vectors.
    Any null vectors are mapped to a uniformly picked unit vector.

    Parameters
    ----------
    v: array, shape (a1, a2, ..., d)
        Cartesian vectors, with last axis indexing the dimension.

    Returns
    -------
    v_new: array, shape of v
    '''
    if v.size == 0:
        return v
    mag = vector_mag(v)
    v_new = v.copy()
    v_new[mag == 0.0] = sphere_pick(v.shape[-1], (mag == 0.0).sum())
    v_new[mag > 0.0] /= mag[mag > 0.0][..., np.newaxis]
    return v_new


def vector_perp(v):
    '''
    Returns vectors perpendicular to 2-dimensional vectors.
    If an input vector has components (x, y), the output vector has
    components (x, -y).

    Parameters
    ----------
    v: array, shape (a1, a2, ..., 2)

    Returns
    -------
    v_perp: array, shape of v
    '''
    if v.shape[-1] != 2:
        raise Exception('Can only define a unique perpendicular vector in 2d')
    v_perp = np.empty_like(v)
    v_perp[..., 0] = v[:, 1]
    v_perp[..., 1] = -v[:, 0]
    return v_perp


# Coordinate system transformations

def polar_to_cart(arr_p):
    '''
    Convert and return polar vectors in their cartesian representation.

    Parameters
    ----------
    arr_p: array, shape (a1, a2, ..., d)
        Polar vectors, with last axis indexing the dimension,
        using (radius, inclination, azimuth) convention.

    Returns
    -------
    arr_c: array, shape of arr_p
        Cartesian vectors.
    '''
    if arr_p.shape[-1] == 1:
        arr_c = arr_p.copy()
    elif arr_p.shape[-1] == 2:
        arr_c = np.empty_like(arr_p)
        arr_c[..., 0] = arr_p[..., 0] * np.cos(arr_p[..., 1])
        arr_c[..., 1] = arr_p[..., 0] * np.sin(arr_p[..., 1])
    elif arr_p.shape[-1] == 3:
        arr_c = np.empty_like(arr_p)
        arr_c[..., 0] = arr_p[..., 0] * np.sin(
            arr_p[..., 1]) * np.cos(arr_p[..., 2])
        arr_c[..., 1] = arr_p[..., 0] * np.sin(
            arr_p[..., 1]) * np.sin(arr_p[..., 2])
        arr_c[..., 2] = arr_p[..., 0] * np.cos(arr_p[..., 1])
    else:
        raise Exception('Invalid vector for polar representation')
    return arr_c


def cart_to_polar(arr_c):
    '''
    Convert and return cartesian vectors in their polar representation.

    Parameters
    ----------
    arr_c: array, shape (a1, a2, ..., d)
        Cartesian vectors, with last axis indexing the dimension.

    Returns
    -------
    arr_p: array, shape of arr_c
        Polar vectors, using (radius, inclination, azimuth) convention.
    '''
    if arr_c.shape[-1] == 1:
        arr_p = arr_c.copy()
    elif arr_c.shape[-1] == 2:
        arr_p = np.empty_like(arr_c)
        arr_p[..., 0] = vector_mag(arr_c)
        arr_p[..., 1] = np.arctan2(arr_c[..., 1], arr_c[..., 0])
    elif arr_c.shape[-1] == 3:
        arr_p = np.empty_like(arr_c)
        arr_p[..., 0] = vector_mag(arr_c)
        arr_p[..., 1] = np.arccos(arr_c[..., 2] / arr_p[..., 0])
        arr_p[..., 2] = np.arctan2(arr_c[..., 1], arr_c[..., 0])
    else:
        raise Exception('Invalid vector for polar representation')
    return arr_p


def sphere_pick_polar(d, n=1):
    '''
    Returns polar vectors uniformly picked on the unit sphere in a space
    with an arbitrary number of dimensions.

    Parameters
    ----------
    d: float
        The number of dimensions of the space in which the sphere lives.
    n: integer
        Number of samples to pick.

    Returns
    -------
    r: array, shape (n, d)
        Sample vectors.
    '''
    a = np.empty([n, d])
    if d == 1:
        a[:, 0] = np.random.randint(2, size=n) * 2 - 1
    elif d == 2:
        a[:, 0] = 1.0
        a[:, 1] = np.random.uniform(-np.pi, +np.pi, n)
    elif d == 3:
        u, v = np.random.uniform(0.0, 1.0, (2, n))
        a[:, 0] = 1.0
        a[:, 1] = np.arccos(2.0 * v - 1.0)
        a[:, 2] = 2.0 * np.pi * u
    else:
        raise Exception('Invalid vector for polar representation')
    return a


def sphere_pick(d, n=1):
    '''
    Returns cartesian vectors uniformly picked on the unit sphere in a space
    with an arbitrary number of dimensions.

    Parameters
    ----------
    d: float
        The number of dimensions of the space in which the sphere lives.
    n: integer
        Number of samples to pick.

    Returns
    -------
    r: array, shape (n, d)
        Sample cartesian vectors.
    '''
    return polar_to_cart(sphere_pick_polar(d, n))


def rejection_pick(L, n, d, valid):
    '''
    Returns cartesian vectors uniformly picked in a space with an arbitrary
    number of dimensions, which is fully enclosed by a cube of finite length,
    using a supplied function which should evaluate whether a picked point lies
    within this space.

    The picking is done by rejection sampling in the cube.

    Parameters
    ----------
    L: float
        Side length of the enclosing cube.
    n: integer
        Number of points to return
    d: integer
        The number of dimensions of the space

    Returns
    -------
    r: array, shape (n, d)
        Sample cartesian vectors
    '''
    rs = []
    while len(rs) < n:
        r = np.random.uniform(-L / 2.0, L / 2.0, size=d)
        if valid(r):
            rs.append(r)
    return np.array(rs)


def ball_pick(n, d):
    '''
    Returns cartesian vectors uniformly picked on the unit ball in an arbitrary
    number of dimensions.

    The unit ball is the space enclosed by the unit sphere.

    The picking is done by rejection sampling in the unit cube.

    In 3-dimensional space, the fraction `\pi / 6 \sim 0.52` points are valid.

    Parameters
    ----------
    n: integer
        Number of points to return.
    d: integer
        Number of dimensions of the space in which the ball lives

    Returns
    -------
    r: array, shape (n, d)
        Sample cartesian vectors.
    '''
    def valid(r):
        return vector_mag_sq(r) < 1.0
    return rejection_pick(L=2.0, n=n, d=d, valid=valid)


def disk_pick_polar(n=1):
    '''
    Returns polar vectors uniformly picked on the unit disk.
    The unit disk is the space enclosed by the unit circle.

    Parameters
    ----------
    n: integer
        Number of points to return.

    Returns
    -------
    r: array, shape (n, 2)
        Sample vectors.
    '''
    a = np.zeros([n, 2], dtype=np.float)
    a[:, 0] = np.sqrt(np.random.uniform(size=n))
    a[:, 1] = np.random.uniform(0.0, 2.0 * np.pi, size=n)
    return a


def disk_pick(n=1):
    '''
    Returns cartesian vectors uniformly picked on the unit disk.
    The unit disk is the space enclosed by the unit circle.

    Parameters
    ----------
    n: integer
        Number of samples to pick.

    Returns
    -------
    r: array, shape (n, 2)
        Sample vectors.
    '''
    return polar_to_cart(disk_pick_polar(n))
