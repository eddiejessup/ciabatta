"""
Areas, volumes, and distances for geometric objects.
"""
from __future__ import print_function, division
import numpy as np
import scipy.special
from ciabatta import vector
from ciabatta._geom_numerics import (sphere_intersection,
                                     spherocylinder_intersection)


SMALL = 1e-10


def sphere_volume(R, n):
    """Return the volume of a sphere in an arbitrary number of dimensions.

    Parameters
    ----------
    R: array-like
        Radius.
    n: array-like
        The number of dimensions of the space in which the sphere lives.

    Returns
    -------
    V: array-like
        Volume.
    """
    return ((np.pi ** (n / 2.0)) / scipy.special.gamma(n / 2.0 + 1)) * R ** n


def sphere_radius(V, n):
    """Return the radius of a sphere in an arbitrary number of dimensions.

    Parameters
    ----------
    V: array-like
        Volume.
    n: array-like
        The number of dimensions of the space in which the sphere lives.

    Returns
    -------
    R: array-like
        Radius.
    """
    return (((scipy.special.gamma(n / 2.0 + 1.0) * V) ** (1.0 / n)) /
            np.sqrt(np.pi))


def sphere_area(R, n):
    """Return the surface area of a sphere in an arbitrary number of dimensions.

    Note that in 2d this will return what is usually called a circle's
    circumference, not what is usually called its area
    (which is in fact its volume).

    Parameters
    ----------
    R: array-like
        Radius.
    n: array-like
        The number of dimensions of the space in which the sphere lives.

    Returns
    -------
    A: array-like
        Surface area.
    """
    return (n / R) * sphere_volume(R, n)


def ellipsoid_volume(a, b, c):
    """Return the volume of an ellipsoid.

    Parameters
    ----------
    a, b, c: array-like
        Length of the semi-axes.
        This is like a generalisation of the radius of a sphere.

    Returns
    -------
    V: array-like
        Volume.
    """
    return (4.0 / 3.0) * np.pi * a * b * c


def cylinder_volume(R, l):
    """Return the volume of a cylinder.

    Parameters
    ----------
    R: array-like
        Radius.
    l: array-like
        Length.

    Returns
    -------
    V: array-like
        Volume.
    """
    # Remember, the volume of a sphere in 2d is what's usually called its area
    return sphere_volume(R, 2) * l


def cylinder_area(R, l):
    """Return the area of a cylinder.

    Parameters
    ----------
    R: array-like
        Radius.
    l: array-like
        Length.

    Returns
    -------
    A: array-like
        Surface area.
    """
    return sphere_area(R, 2) * l


def spherocylinder_volume(R, l):
    """Return the volume of a
    [spherocylinder](http://en.wikipedia.org/wiki/Capsule_(geometry)).

    Parameters
    ----------
    R: array-like
        Radius of the hemispheres and cylinder sections.
    l: array-like
        Length of the cylinder section.

    Returns
    -------
    V: array-like
        Volume.
    """
    return sphere_volume(R, 3) + cylinder_volume(R, l)


def spherocylinder_area(R, l):
    """Return the surface area of a
    [spherocylinder](http://en.wikipedia.org/wiki/Capsule_(geometry)).

    Parameters
    ----------
    R: array-like
        Radius of the hemispheres and cylinder sections.
    l: array-like
        Length of the cylinder section.

    Returns
    -------
    A: array-like
        Surface area.
    """
    return sphere_area(R, 3) + cylinder_area(R, l)


def spherocylinder_radius(V, l):
    """Return the radius of a
    [spherocylinder](http://en.wikipedia.org/wiki/Capsule_(geometry)).

    Parameters
    ----------
    V: float
        Volume.
    l: float
        Length of the cylinder section.

    Returns
    -------
    R: float
        Radius.
    """
    return np.roots([4.0 / 3.0, l, 0, -V / np.pi])[-1].real


def spherocylinder_aspect_ratio(l, R):
    """Return the aspect ratio of a spherocylinder,

    Parameters
    ----------
    l: float
        Length of the cylinder section.
    R: float
        Radius of the hemispheres and cylinder sections.

    Returns
    -------
    ar: float
        Aspect ratio.
        This is defined as the ratio of length including hemisphere
        sections, to radius.
    """
    return 1.0 + l / (2.0 * R)


def spherocylinder_radius_for_aspect(V, ar):
    """Return the radius of a spherocylinder with a given volume and
    aspect ratio.

    Parameters
    ----------
    V: float
        Volume.
    ar: float
        Aspect ratio.
        This is defined as the ratio of length including hemisphere
        sections, to radius.

    Returns
    -------
    R: float
        Radius.
    """
    return (V / (2.0 * np.pi * (a - (1.0 / 3.0)))) ** (1.0 / 3.0)


def spheres_sep(ar, aR, br, bR):
    """Return the separation distance between two spheres.

    Parameters
    ----------
    ar, br: array-like, shape (n,) in n dimensions
        Coordinates of the centres of the spheres `a` and `b`.
    aR, bR: float
        Radiuses of the spheres `a` and `b`.

    Returns
    -------
    d: float
        Separation distance.
        A negative value means the spheres intersect each other.
    """
    return vector.vector_mag(ar - br) - (aR + bR)


def spheres_intersect(ar, aR, br, bR):
    """Return whether or not two spheres intersect each other.

    Parameters
    ----------
    ar, br: array-like, shape (n,) in n dimensions
        Coordinates of the centres of the spheres `a` and `b`.
    aR, bR: float
        Radiuses of the spheres `a` and `b`.

    Returns
    -------
    intersecting: boolean
        True if the spheres intersect.
    """
    return vector.vector_mag_sq(ar - br) < (aR + bR) ** 2


def point_seg_sep(ar, br1, br2):
    """Return the minimum separation vector between a point and a line segment,
    in 3 dimensions.

    Parameters
    ----------
    ar: array-like, shape (3,)
        Coordinates of a point.
    br1, br2: array-like, shape (3,)
        Coordinates for the points of a line segment

    Returns
    -------
    sep: float array, shape (3,)
        Separation vector between point and line segment.
    """
    v = br2 - br1
    w = ar - br1

    c1 = np.dot(w, v)
    if c1 <= 0.0:
        return ar - br1

    c2 = np.sum(np.square(v))
    if c2 <= c1:
        return ar - br2

    b = c1 / c2
    bc = br1 + b * v
    return ar - bc


def point_seg_sep_sq(ar, br1, br2):
    """Return the squared minimum separation distance between a point and a
    line segment, in 3 dimensions.

    Parameters
    ----------
    ar: array-like, shape (3,)
        Coordinates of a point.
    br1, br2: array-like, shape (3,)
        Coordinates for the points of a line segment

    Returns
    -------
    d_sq: float
        Squared separation distance between point and line segment.
    """
    v = br2 - br1
    w = ar - br1

    c1 = np.dot(w, v)
    if c1 <= 0.0:
        return np.sum(np.square(ar - br1))

    c2 = np.sum(np.square(v))
    if c2 <= c1:
        return np.sum(np.square(ar - br2))

    b = c1 / c2
    bc = br1 + b * v
    return np.sum(np.square(ar - bc))
