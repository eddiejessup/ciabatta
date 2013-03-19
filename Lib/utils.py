import os
import numpy as np
import scipy.special

# File IO

def makedirs_safe(dirname):
    if os.path.isdir(dirname):
        s = input('%s exists, overwrite? (y/n) ' % dirname)
        if s != 'y': raise Exception
    else:
        os.makedirs(dirname)

def makedirs_soft(dirname):
    if not os.path.isdir(dirname): os.makedirs(dirname)

# Strings

def suffix_remove(s, suffix):
    if s.endswith(suffix): return s[:-len(suffix)]
    else: return s

# Index- and real-space conversion and wrapping

def r_to_i(r, L, dx):
    return np.asarray((r + L / 2.0) / dx, dtype=np.int)

def i_to_r(i, L, dx):
    return -L / 2.0 + (i + 0.5) * dx

def wrap_real(L, L_half, r):
    if r > L_half: r -= L
    elif r < -L_half: r += L
    return r

def wrap(M, i):
    if i >= M: i -= M
    elif i < 0: i += M
    return i

def wrap_inc(M, i):
    return i + 1 if i < M - 1 else 0

def wrap_dec(M, i):
    return i - 1 if i > 0 else M - 1

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
    if v.size == 0: return v
    mag = vector_mag(v)
    v_new = v.copy()
    v_new /= mag[..., np.newaxis]
    return v_new

def vector_unit_nullnull(v):
    ''' Array of cartesian vectors into unit vectors.
    If null vector encountered, leave as null.
    Assumes last index is that of the vector component. '''
    if v.size == 0: return v
    mag = vector_mag(v)
    v_new = v.copy()
    v_new[mag > 0.0] /= mag[mag > 0.0][..., np.newaxis]
    return v_new

def vector_unit_nullrand(v):
    ''' Array of cartesian vectors into unit vectors.
    If null vector encountered, pick new random unit vector.
    Assumes last index is that of the vector component. '''
    if v.size == 0: return v
    mag = vector_mag(v)
    v_new = v.copy()
    v_new[mag == 0.0] = sphere_pick(v.shape[-1], (mag == 0.0).sum())
    v_new[mag > 0.0] /= mag[mag > 0.0][..., np.newaxis]
    return v_new

def vector_angle(a, b):
    cos_theta = np.sum(a * b, -1) / (vector_mag(a) * vector_mag(b))
    theta = np.empty_like(cos_theta)
    try:
        theta[np.abs(cos_theta) <= 1.0] = np.arccos(cos_theta[np.abs(cos_theta) <= 1.0])
    except IndexError:
        if np.abs(cos_theta) <= 1.0: theta = np.arccos(cos_theta)
        elif np.dot(a, b) > 0.0: theta = 0.0
        else: theta = np.pi
    else:
        for i in np.where(np.abs(cos_theta) > 1.0)[0]:
            if np.dot(a[i], b[i]) > 0.0: theta[i] = 0.0
            else: theta[i] = np.pi
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
        arr_c[..., 0] = arr_p[..., 0] * np.sin(arr_p[..., 1]) * np.cos(arr_p[..., 2])
        arr_c[..., 1] = arr_p[..., 0] * np.sin(arr_p[..., 1]) * np.sin(arr_p[..., 2])
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
    R = np.zeros((len(th), 2, 2), dtype=np.float)
    R[:, 0, :] = np.array((c, -s)).T
    R[:, 1, :] = np.array((s, c)).T
    return R

def R_rot_3d(th):
    sx, sy, sz = np.sin(th).T
    cx, cy, cz = np.cos(th).T
    R = np.zeros((len(th), 3, 3), dtype=np.float)
    R[:, 0, :] = np.array((cy*cz, -cy*sz, sy)).T
    R[:, 1, :] = np.array((sx*sy*cz + cx*sz, -sx*sy*sz + cx*cz, -sx*cy)).T
    R[:, 2, :] = np.array((-cx*sy*cz + sx*sz, cx*sy*sz + sx*cz, cx*cy)).T
    return R

def R_rot(th):
    if th.shape[-1] == 2: return R_rot_2d(th)
    elif th.shape[-1] == 3: return R_rot_3d(th)
    else: raise Exception('Rotation matrix not implemented in this dimension')

def rotate(a, th):
    a_rot = np.zeros_like(a)
    R = R_rot(th)
    for i in range(len(a)):
        a_rot[i] = R[i].dot(a[i])
    return a_rot

# Diffusion

def rot_diff(v, D, dt):
    if D * dt == 0.0: return v.copy()
    angles = int(round(v.shape[-1] * (v.shape[-1] - 1) / 2.0))
    return rotate(v, np.sqrt(2.0 * D * dt) * np.random.standard_normal((len(v), angles)))

def calc_D_rot(v1, v2, dt):
    if dt == 0.0: return np.nan
    return np.mean(np.square(vector_angle(v1, v2))) / (2.0 * (v1.shape[-1] - 1) * dt)

def diff(r, D, dt):
    if D * dt == 0.0: return r.copy()
    return r + np.sqrt(2.0 * D * dt) * np.random.standard_normal(r.shape)

def calc_D_vector(r1, r2, dt):
    if dt == 0.0: return 2 * (np.array(r1.shape[1] * (np.nan,)),)
    D = np.square(r1 - r2) / (2.0 * dt)
    return np.mean(D, axis=0), np.std(D, axis=0) / np.sqrt(len(r1))

def calc_D(r1, r2, dt):
    D_vect, D_vect_err = calc_D_vector(r1, r2, dt)
    return np.mean(D_vect), np.sqrt(np.sum(np.square(D_vect_err)))

def calc_v_drift(r1, r2, dt):
    if dt == 0.0: return 2 * (np.array(r1.shape[1] * (np.nan,)),)
    v = (r1 - r2) / dt
    return np.mean(v, axis=0), np.std(v, axis=0) / np.sqrt(len(v))

# Numpy arrays

def extend_array(a, n):
    if n < 1: raise Exception('Extend factor >= 1')
    M_new = a.shape[0] * n
    a_new = np.zeros(a.ndim * [M_new], dtype=a.dtype)
    for x_new in range(M_new):
        for y_new in range(M_new):
            a_new[x_new, y_new] = a[x_new // n, y_new // n]
    return a_new

def field_subset(f, inds, rank=0):
    f_dim_space = f.ndim - rank
    if inds.ndim > 2:
        raise Exception('Too many dimensions in indices array')
    if inds.ndim == 1:
        if f_dim_space == 1: return f[inds]
        else: raise Exception('Indices array is 1d but field is not')
    if inds.shape[1] != f_dim_space:
        raise Exception('Indices and field dimensions do not match')
    # It's magic, don't touch it!
    return f[tuple([inds[:, i] for i in range(inds.shape[1])])]

# Spheres

def sphere_intersect(r_1, R_1, r_2, R_2):
    return vector_mag_sq(r_1 - r_2) < (R_1 + R_2) ** 2

def sphere_pack(R, n, pf):
    if not 0.0 < R < 1.0:
        raise Exception('Require 0 < sphere radius < 1')
    if n == 2 and pf > np.pi / np.sqrt(12):
        raise Exception('Cannot achieve packing fraction')
    elif n == 3 and pf > np.pi / np.sqrt(18):
        raise Exception('Cannot achieve packing fraction')
    else:
        print('Warning: No guarantee the requested packing fraction is '
              'achievable')
    if not 0.0 <= pf < 1.0:
        raise Exception('Require 0 < packing fraction < 1')

    m = int(round(pf / sphere_volume(R, n)))
    rs = np.zeros([m, n], dtype=np.float)
    lim = 0.5 - R
    for i in range(m):
        while True:
            rs[i] = np.random.uniform(-lim, lim, n)
            valid = True
            for r in rs[:i]:
                if sphere_intersect(rs[i], R, r, R):
                    valid = False
                    break
            if valid: break
    return rs

def sphere_volume(R, n):
    return ((np.pi ** (n / 2.0)) / scipy.special.gamma(n / 2.0 + 1)) * R ** n

def sphere_area(R, n):
    '''
    Bounding area of a sphere of radius R in n-dimensional space.
    NOTE: In 2d this will return a circle's circumference, not what is commonly
    referred to as its area.
    '''
    return (n / R) * volume(R, n)
