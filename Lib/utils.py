import os
import subprocess
import numpy as np


def get_git_hash():
    '''
    Return the shortened git SHA of the current commit.
    '''
    cmd = ["git", "rev-parse", "--short", "HEAD"]
    return subprocess.check_output(cmd).strip()


# File IO

def makedirs_safe(dirname):
    '''
    Make a directory, prompting the user if it already exists.
    '''
    if os.path.isdir(dirname):
        s = input('%s exists, overwrite? (y/n) ' % dirname)
        if s != 'y':
            raise Exception
    else:
        os.makedirs(dirname)


def makedirs_soft(dirname):
    '''
    Make a directory, if it doesn't already exist.
    Otherwise do nothing.
    '''
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


# Index- and real-space conversion and wrapping

def r_to_i(r, L, dx):
    '''
    Return closest indices on a square lattice of vectors in continuous space.

    Parameters
    ----------
    r: float array, shape (a1, a2, ..., d)
        Cartesian vectors, with last axis indexing the dimension.
        It's assumed that all components of the vector lie within (± L / 2).
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
        lie within (± (L / dx) / 2).
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


# Vectors

def vector_mag_sq(v):
    '''
    Return the squared magnitude of vectors v.

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
    Return the magnitude of vectors v.

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
    Return unit vectors of input vectors.
    Any null vectors in v raise an Exception

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
    Return unit vectors of input vectors.
    Any null vectors in v are mapped again to null vectors.

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
    Return unit vectors of input vectors.
    Any null vectors in v are mapped to randomly picked unit vectors.

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


def vector_angle(a, b):
    '''
    Return angles between two sets of vectors.

    Parameters
    ----------
    a, b: array, shape (a1, a2, ..., d)
        Cartesian vectors, with last axis indexing the dimension.

    Returns
    -------
    theta: array, shape (a1, a2, ...)
        Angles between a and b.
    '''
    cos_theta = np.sum(a * b, -1) / (vector_mag(a) * vector_mag(b))
    theta = np.empty_like(cos_theta)
    try:
        theta[np.abs(cos_theta) <= 1.0] = np.arccos(
            cos_theta[np.abs(cos_theta) <= 1.0])
    except IndexError:
        if np.abs(cos_theta) <= 1.0:
            theta = np.arccos(cos_theta)
        elif np.dot(a, b) > 0.0:
            theta = 0.0
        else:
            theta = np.pi
    else:
        for i in np.where(np.abs(cos_theta) > 1.0)[0]:
            if np.dot(a[i], b[i]) > 0.0:
                theta[i] = 0.0
            else:
                theta[i] = np.pi
    return theta


def vector_perp(v):
    '''
    Return vectors perpendicular to 2-dimensional vectors.
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


# Point picking

def sphere_pick_polar(d, n=1):
    '''
    Return polar vectors randomly picked on the unit n-sphere.

    Parameters
    ----------
    d: float
        Dimensionality of the sphere.
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
    Return cartesian vectors randomly picked on the unit n-sphere.

    Parameters
    ----------
    d: float
        Dimensionality of the sphere.
    n: integer
        Number of samples to pick.

    Returns
    -------
    r: array, shape (n, d)
        Sample cartesian vectors.
    '''
    return polar_to_cart(sphere_pick_polar(d, n))


def disk_pick_polar(n=1):
    '''
    Return polar vectors randomly picked on the 2-dimensional unit disk.

    Parameters
    ----------
    n: integer
        Number of samples to pick.

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
    Return cartesian vectors randomly picked on the 2-dimensional unit disk.

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


# Rotations

def R_rot_2d(th):
    '''
    Return a 2-dimensional rotation matrix.

    Parameters
    ----------
    th: array, shape (n, 1)
        Angles about which to rotate.

    Returns
    -------
    R: array, shape (n, 2, 2)
    '''
    s, = np.sin(th).T
    c, = np.cos(th).T
    R = np.empty((len(th), 2, 2), dtype=np.float)

    R[:, 0, 0] = c
    R[:, 0, 1] = -s

    R[:, 1, 0] = s
    R[:, 1, 1] = c

    return R


def R_rot_3d(th):
    '''
    Return a 3-dimensional rotation matrix.

    Parameters
    ----------
    th: array, shape (n, 3)
        Angles about which to rotate along each axis.

    Returns
    -------
    R: array, shape (n, 3, 3)
    '''
    sx, sy, sz = np.sin(th).T
    cx, cy, cz = np.cos(th).T
    R = np.empty((len(th), 3, 3), dtype=np.float)

    R[:, 0, 0] = cy * cz
    R[:, 0, 1] = -cy * sz
    R[:, 0, 2] = sy

    R[:, 1, 0] = sx * sy * cz + cx * sz
    R[:, 1, 1] = -sx * sy * sz + cx * cz
    R[:, 1, 2] = -sx * cy

    R[:, 2, 0] = -cx * sy * cz + sx * sz
    R[:, 2, 1] = cx * sy * sz + sx * cz
    R[:, 2, 2] = cx * cy
    return R


def R_rot(th):
    '''
    Return a rotation matrix.

    Parameters
    ----------
    th: array, shape (n, m)
        Angles by which to rotate about each m rotational degree of freedom
        (m=1 in 2 dimensions, m=3 in 3 dimensions).

    Returns
    -------
    R: array, shape (n, m, m)
    '''
    try:
        dof = th.shape[-1]
    # If th is a python float
    except AttributeError:
        dof = 1
        th = np.array([th])
    except IndexError:
        dof = 1
        th = np.array([th])
    # If th is a numpy float, i.e. 0d array
    if dof == 1:
        return R_rot_2d(th)
    elif dof == 3:
        return R_rot_3d(th)
    else:
        raise Exception('Rotation matrix not implemented in this dimension')


def rotate(a, th):
    '''
    Return cartesian vectors, after rotation by specified angles about
    each degree of freedom.

    Parameters
    ----------
    a: array, shape (n, d)
        Input d-dimensional cartesian vectors, left unchanged.
    th: array, shape (n, m)
        Angles by which to rotate about each m rotational degree of freedom
        (m=1 in 2 dimensions, m=3 in 3 dimensions).

    Returns
    -------
    ar: array, shape of a
        Rotated cartesian vectors.
    '''
    return np.sum(a[..., np.newaxis] * R_rot(th), axis=-2)


# Diffusion

def rot_diff(v, D, dt):
    '''
    Return cartesian velocity vectors, after applying rotational diffusion.

    Parameters
    ----------
    v: array, shape (n, d)
        Cartesian velocity vectors in d dimensions, left unchanged.
    D: array, shape (n)
        Rotational diffusion constant for each vector.
    dt: float
        Time interval over which rotational diffusion acts.

    Returns
    -------
    vr: array, shape of v
        Velocity vectors after rotational diffusion is applied.
    '''
    # Account for possibility of D being an array
    try:
        D = D[:, np.newaxis]
    # If D is a python float
    except TypeError:
        pass
    # If D is a numpy float, i.e. 0-d array
    except IndexError:
        pass
    dim = v.shape[-1]
    dof = dim * (dim - 1) // 2
    th = np.sqrt(2.0 * D * dt) * np.random.standard_normal((len(v), dof))
    return rotate(v, th)


def diff(r, D, dt):
    '''
    Return cartesian position vectors, after applying translational diffusion.

    Parameters
    ----------
    r: array, shape (n, d)
        Cartesian position vectors in d dimensions, left unchanged.
    D: array, shape (n)
        Translational diffusion constant for each vector.
    dt: float
        Time interval over which translational diffusion acts.

    Returns
    -------
    rr: array, shape of r
        Velocity vectors after translational diffusion is applied.
    '''
    if dt == 0.0:
        return r.copy()
    return r + np.sqrt(2.0 * D * dt) * np.random.standard_normal(r.shape)


# Arrays

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
