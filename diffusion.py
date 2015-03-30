from __future__ import print_function, division
import numpy as np
from ciabatta import rotation


def rot_diff(v, D, dt):
    '''
    Returns cartesian velocity vectors, after applying rotational diffusion.

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
    return rotation.rotate(v, th)


def diff(r, D, dt):
    '''
    Returns cartesian position vectors, after applying translational diffusion.

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
