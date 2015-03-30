from __future__ import print_function, division
import numpy as np


def R_rot_2d(th):
    '''
    Returns a 2-dimensional rotation matrix.

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
    Returns a 3-dimensional rotation matrix.

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
    Returns a rotation matrix.

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
    Returns cartesian vectors, after rotation by specified angles about
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
