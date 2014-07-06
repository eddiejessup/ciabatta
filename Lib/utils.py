
import os
import subprocess
import numpy as np


def get_git_hash():
    cmd = ["git", "rev-parse", "--short", "HEAD"]
    return subprocess.check_output(cmd).strip()


# File IO

def makedirs_safe(dirname):
    if os.path.isdir(dirname):
        s = input('%s exists, overwrite? (y/n) ' % dirname)
        if s != 'y':
            raise Exception
    else:
        os.makedirs(dirname)


def makedirs_soft(dirname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


# Index- and real-space conversion and wrapping

def r_to_i(r, L, dx):
    return np.asarray((r + L / 2.0) / dx, dtype=np.int)


def i_to_r(i, L, dx):
    return -L / 2.0 + (i + 0.5) * dx


# Centre of mass

def rms_com(r, L):
    ''' RMS distance of array of cartesian vectors r from their
    centre-of-mass vector.
    Assumes last index is that of the vector component. '''
    return rms(r, com(r, L), L)


def com(r, L):
    ''' Centre-of-mass vector of array of cartesian vectors r in
    a periodic system with period L.
    Assumes last index is that of the vector component. '''
    r_com = np.zeros([r.shape[-1]], dtype=np.float)
    for i_dim in range(len(r_com)):
        x_0 = r[:, i_dim]
        x_av = np.mean(x_0)
        rms_min = rms(x_0, x_av, L)
        steps_0 = 4
        steps = float(steps_0)
        while True:
            for x_base in np.arange(-L / 2.0 + L / steps, L / 2.0, L / steps):
                x_new = x_0.copy()
                x_new[np.where(x_new < x_base)] += L
                x_av_new = np.mean(x_new)
                if x_av_new > L / 2.0:
                    x_av_new -= L
                rms_new = rms(x_new, x_av_new, L)
                if rms_new < rms_min:
                    x_av = x_av_new
                    rms_min = rms_new

            if (rms(x_0, 0.99 * x_av, L) < rms_min or
                    rms(x_0, 1.01 * x_av, L) < rms_min):
                steps *= 2
                print('Recalculating c.o.m. with steps=%i' % steps)
            else:
                r_com[i_dim] = x_av
                break
    return r_com


def rms(r, r_0, L):
    ''' RMS distance of array of cartesian vectors r from point r_0.
    Assumes last index is that of the vector component (x, [y, z, ...]). '''
    r_sep_sq = np.minimum((r - r_0) ** 2, (L - np.abs(r - r_0)) ** 2)
    return np.sqrt(np.mean(np.sum(r_sep_sq, r_sep_sq.ndim - 1)))


# Vectors

def vector_mag_sq(v):
    ''' Squared magnitude of array of cartesian vectors v.
    Assumes last index is that of the vector component. '''
    return np.square(v).sum(axis=-1)


def vector_mag(v):
    ''' Magnitude of array of cartesian vectors v.
    Assumes last index is that of the vector component. '''
    return np.sqrt(vector_mag_sq(v))


def vector_unit_nonull(v):
    ''' Array of cartesian vectors v into unit vectors.
    If null vector encountered, raise exception.
    Assumes last index is that of the vector component. '''
    if v.size == 0:
        return v
    return v / vector_mag(v)[..., np.newaxis]


def vector_unit_nullnull(v):
    ''' Array of cartesian vectors into unit vectors.
    If null vector encountered, leave as null.
    Assumes last index is that of the vector component. '''
    if v.size == 0:
        return v
    mag = vector_mag(v)
    v_new = v.copy()
    v_new[mag > 0.0] /= mag[mag > 0.0][..., np.newaxis]
    return v_new


def vector_unit_nullrand(v):
    ''' Array of cartesian vectors into unit vectors.
    If null vector encountered, pick new random unit vector.
    Assumes last index is that of the vector component. '''
    if v.size == 0:
        return v
    mag = vector_mag(v)
    v_new = v.copy()
    v_new[mag == 0.0] = sphere_pick(v.shape[-1], (mag == 0.0).sum())
    v_new[mag > 0.0] /= mag[mag > 0.0][..., np.newaxis]
    return v_new


def vector_angle(a, b):
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
    ''' Vector perpendicular to 2D vector v '''
    if v.shape[-1] != 2:
        raise Exception('Can only define a unique perpendicular vector in 2d')
    v_perp = np.empty_like(v)
    v_perp[..., 0] = v[:, 1]
    v_perp[..., 1] = -v[:, 0]
    return v_perp


# Coordinate system transformations

def polar_to_cart(arr_p):
    ''' Array of vectors arr_c corresponding to cartesian
    representation of array of polar vectors arr_p.
    Assumes last index is that of the vector component.
    In 3d assumes (radius, inclination, azimuth) convention. '''
    if arr_p.shape[-1] == 1:
        arr_c = arr_p.copy()
    elif arr_p.shape[-1] == 2:
        arr_c = np.zeros_like(arr_p)
        arr_c[..., 0] = arr_p[..., 0] * np.cos(arr_p[..., 1])
        arr_c[..., 1] = arr_p[..., 0] * np.sin(arr_p[..., 1])
    elif arr_p.shape[-1] == 3:
        arr_c = np.zeros_like(arr_p)
        arr_c[..., 0] = arr_p[..., 0] * np.sin(
            arr_p[..., 1]) * np.cos(arr_p[..., 2])
        arr_c[..., 1] = arr_p[..., 0] * np.sin(
            arr_p[..., 1]) * np.sin(arr_p[..., 2])
        arr_c[..., 2] = arr_p[..., 0] * np.cos(arr_p[..., 1])
    else:
        raise Exception('Invalid vector for polar representation')
    return arr_c


def cart_to_polar(arr_c):
    ''' Array of vectors arr_p corresponding to polar representation
    of array of cartesian vectors arr_c.
    Assumes last index is that of the vector component.
    In 3d uses (radius, inclination, azimuth) convention. '''
    if arr_c.shape[-1] == 1:
        arr_p = arr_c.copy()
    elif arr_c.shape[-1] == 2:
        arr_p = np.zeros_like(arr_c)
        arr_p[..., 0] = vector_mag(arr_c)
        arr_p[..., 1] = np.arctan2(arr_c[..., 1], arr_c[..., 0])
    elif arr_c.shape[-1] == 3:
        arr_p = np.zeros_like(arr_c)
        arr_p[..., 0] = vector_mag(arr_c)
        arr_p[..., 1] = np.arccos(arr_c[..., 2] / arr_p[..., 0])
        arr_p[..., 2] = np.arctan2(arr_c[..., 1], arr_c[..., 0])
    else:
        raise Exception('Invalid vector for polar representation')
    return arr_p


# Point picking

def sphere_pick_polar(d, n=1):
    ''' In 3d uses (radius, inclination, azimuth) convention '''
    a = np.zeros([n, d], dtype=np.float)
    if d == 1:
        a[:, 0] = np.random.randint(2, size=n) * 2 - 1
    elif d == 2:
        a[:, 0] = 1.0
        a[:, 1] = np.random.uniform(-np.pi, +np.pi, n)
    elif d == 3:
        # Note, (r, theta, phi) notation
        u, v = np.random.uniform(0.0, 1.0, (2, n))
        a[:, 0] = 1.0
        a[:, 1] = np.arccos(2.0 * v - 1.0)
        a[:, 2] = 2.0 * np.pi * u
    else:
        raise Exception('Invalid vector for polar representation')
    return a


def sphere_pick(d, n=1):
    return polar_to_cart(sphere_pick_polar(d, n))


def disk_pick_polar(n=1):
    a = np.zeros([n, 2], dtype=np.float)
    a[:, 0] = np.sqrt(np.random.uniform(size=n))
    a[:, 1] = np.random.uniform(0.0, 2.0 * np.pi, size=n)
    return a


def disk_pick(n=1):
    return polar_to_cart(disk_pick_polar(n))


# Rotations

def R_rot_2d(th):
    s, = np.sin(th).T
    c, = np.cos(th).T
    R = np.empty((len(th), 2, 2), dtype=np.float)

    R[:, 0, 0] = c
    R[:, 0, 1] = -s

    R[:, 1, 0] = s
    R[:, 1, 1] = c

    return R


def R_rot_3d(th):
    sx, sy, sz = np.sin(th).T
    cx, cy, cz = np.cos(th).T
    R = np.empty((len(th), 3, 3), dtype=np.float)

    R[:, 0, 0] = cy * cz
    R[:, 0, 1] = -cy * sz
    R[:, 0, 1] = sy

    R[:, 1, 0] = sx * sy * cz + cx * sz
    R[:, 1, 1] = -sx * sy * sz + cx * cz
    R[:, 1, 1] = -sx * cy

    R[:, 2, 0] = -cx * sy * cz + sx * sz
    R[:, 2, 1] = cx * sy * sz + sx * cz
    R[:, 2, 1] = cx * cy
    return R


def R_rot(th):
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
    return np.sum(a[..., np.newaxis] * R_rot(th), axis=-2)


# Diffusion

def rot_diff(v, D, dt):
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
    if dt == 0.0:
        return r.copy()
    return r + np.sqrt(2.0 * D * dt) * np.random.standard_normal(r.shape)


# Arrays

def extend_array(a, n):
    a_new = a.copy()
    for d in range(a.ndim):
        a_new = np.repeat(a_new, n, axis=d)
    return a_new


def field_subset(f, inds, rank=0):
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
