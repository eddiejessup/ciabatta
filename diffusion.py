from __future__ import print_function, division
import numpy as np
import scipy.constants
from ciabatta import geom, rotation


def spheroid_xi(p):
    return np.sqrt(np.abs(p ** 2.0 - 1.0)) / p


def perrin_factor(p):
    xi = spheroid_xi(p)
    if p > 1.0:
        return 2.0 * np.arctanh(xi) / xi
    else:
        return 2.0 * np.arctan(xi) / xi


def drag_coeff_sphere(R, eta):
    return 6.0 * np.pi * eta * R


def drag_coeff_rot_sphere(R, eta):
    return 8.0 * np.pi * eta * R ** 3


def drag_coeff_spheroid(a, b, eta):
    p = a / b
    S = perrin_factor(p)
    f_p = 2 * p ** (2.0 / 3.0) / S
    V = geom.ellipsoid_volume(a, b, b)
    R_sph = geom.sphere_radius(V, n=3)
    f_sph = drag_coeff_sphere(eta, R_sph)
    return f_sph * f_p


def drag_coeff_rot_spheroid(a, b, eta):
    p = a / b
    S = perrin_factor(p)
    xi = spheroid_xi(p)
    F_axial = (4.0 / 3.0) * xi ** 2 / (2.0 - (S / p ** 2))
    F_equat = ((4.0 / 3.0) *
               ((1.0 / p) ** 2 - p ** 2) / (2.0 - S * (2.0 - (1.0 / p) ** 2)))
    return F_axial, F_equat


def drag_coeff_spherocylinder(R, l, eta):
    V = geom.spherocylinder_volume(R, l)
    R_sph = geom.sphere_radius(V, n=3)
    return drag_coeff_sphere(R_sph, eta)


def drag_coeff_rot_spherocylinder(R, l, eta):
    V = geom.spherocylinder_volume(R, l)
    R_sph = geom.sphere_radius(V, n=3)
    return drag_coeff_rot_sphere(R_sph, eta)


def stokes_einstein(drag, T):
    """Returns the diffusion coefficient for an object.

    The viscosity of water at 300 K is 0.001 Pa.s,
    which has base units kg / (m.s).

    The diffusion constant might be translational or rotational.
    It depends on which one the drag coefficient represents,
    since the Stokes-Einstein relation is the same for both.

    Parameters
    ----------
    drag:
        The drag coefficient for the object
    T:
        Temperature

    Returns
    -------
    D:
        Diffusion constant.

    """
    return scipy.constants.k * T / drag


def rot_diff(v, D, dt, rng=None):
    """Returns cartesian velocity vectors, after applying rotational diffusion.

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
    """
    if rng is None:
        rng = np.random
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
    th = np.sqrt(2.0 * D * dt) * rng.standard_normal((v.shape[0], dof))
    return rotation.rotate(v, th)


def diff(r, D, dt, rng=None):
    """Returns cartesian position vectors, after applying translational diffusion.

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
    """
    if rng is None:
        rng = np.random
    if dt == 0.0:
        return r.copy()
    return r + np.sqrt(2.0 * D * dt) * rng.standard_normal(r.shape)
