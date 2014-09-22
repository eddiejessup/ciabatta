import numpy as np
import scipy.special
from ciabatta import utils, geom_numerics

SMALL = 1e-10


def sphere_volume(R, n):
    '''
    Returns the volume of an n-dimensional sphere.

    Parameters
    ----------
    R: array-like
        Radius
    n: array-like
        Dimensionality

    Returns
    -------
    V: array-like
        Volume
    '''
    return ((np.pi ** (n / 2.0)) / scipy.special.gamma(n / 2.0 + 1)) * R ** n


def sphere_radius(V, n):
    '''
    Returns the radius of an n-dimensional sphere.

    Parameters
    ----------
    V: array-like
        Volume
    n: array-like
        Dimensionality

    Returns
    -------
    R: array-like
        Radius
    '''
    return (((scipy.special.gamma(n / 2.0 + 1.0) * V) ** (1.0 / n)) /
            np.sqrt(np.pi))


def sphere_area(R, n):
    '''
    Returns the surface area of an n-dimensional sphere.

    Note that in 2d this will return what is usually called a circle's
    circumference, not what is usually called its area
    (which is in fact its volume).

    Parameters
    ----------
    R: array-like
        Radius
    n: array-like
        Dimensionality

    Returns
    -------
    A: array-like
        Surface area
    '''
    return (n / R) * sphere_volume(R, n)


def cylinder_volume(R, l):
    '''
    Returns the volume of a cylinder.

    Parameters
    ----------
    R: array-like
        Radius
    l: array-like
        Length

    Returns
    -------
    V: array-like
        Volume
    '''
    # Remember, the volume of a sphere in 2d is what's usually called its area
    return sphere_volume(R, 2) * l


def cylinder_area(R, l):
    '''
    Returns the volume of a cylinder.

    Parameters
    ----------
    R: array-like
        Radius
    l: array-like
        Length

    Returns
    -------
    A: array-like
        Surface area
    '''
    return sphere_area(R, 2) * l


def capsule_volume(R, l):
    '''
    Returns the volume of a
    [capsule](http://en.wikipedia.org/wiki/Capsule_(geometry)).

    Parameters
    ----------
    R: array-like
        Radius of the capsule hemispheres, and cylinder sections.
    l: array-like
        Length of the cylinder section

    Returns
    -------
    V: array-like
        Volume
    '''
    return sphere_volume(R, 3) + cylinder_volume(R, l)


def capsule_area(R, l):
    '''
    Returns the volume of a
    [capsule](http://en.wikipedia.org/wiki/Capsule_(geometry)).

    Parameters
    ----------
    R: array-like
        Radius of the capsule hemispheres, and cylinder sections.
    l: array-like
        Length of the cylinder section

    Returns
    -------
    A: array-like
        Surface area
    '''
    return sphere_area(R, 3) + cylinder_area(R, l)


def R_of_l(V, l):
    '''
    Returns the radius of a capsule required for a given volume.

    http://en.wikipedia.org/wiki/Capsule_(geometry)

    Parameters
    ----------
    V: float
        Required volume.
    l: float
        Length of the capsule's line segment.

    Returns
    -------
    R: float
        Radius.
    '''
    R = scipy.roots([(4.0 / 3.0) * np.pi, np.pi * l, 0.0, -V])
    R_phys = np.real(R[np.logical_and(np.isreal(R), R > 0.0)])
    if len(R_phys) != 1:
        raise Exception('More or less than one physical radius found')
    return R_phys[0]


def capsule_aspect_ratio(l, R):
    '''
    Returns the aspect ratio of a capsule, defined as the ratio of its length
    including hemisphere sections, to its radius.

    Parameters
    ----------
    l: float
        Length of the capsule's line segment.
    R: float
        Radius.

    Returns
    -------
    ar: float
        Aspect ratio.
    '''
    return 1.0 + l / (2.0 * R)


def spheres_sep(ar, aR, br, bR):
    '''
    Returns the separation distance between two spheres.

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
    '''
    return utils.vector_mag(ar - br) - (aR + bR)


def spheres_intersect(ar, aR, br, bR):
    '''
    Returns whether or not two spheres intersect each other.

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
    '''
    return utils.vector_mag_sq(ar - br) < (aR + bR) ** 2


def sphere_insphere_sep(ar, aR, br, bR):
    '''
    For two spheres `a` and `b`, returns the distance that `b` is inside `a`.

    Parameters
    ----------
    ar, br: array-like, shape (n,) in n dimensions
        Coordinates of the centres of the spheres `a` and `b`.
    aR, bR: float
        Radiuses of the spheres `a` and `b`.

    Returns
    -------
    d: float
        Distance between the surface of `b` and the surface of `a`.
        A negative value means `b` is not entirely inside `a`.
    '''
    return spheres_sep(ar, -aR, br, bR)


def sphere_insphere_intersect(ar, aR, br, bR):
    '''
    For two spheres `a` and `b`, returns True if `b` is entirely
    inside `a`.

    Parameters
    ----------
    ar, br: array-like, shape (n,) in n dimensions
        Coordinates of the centres of the spheres `a` and `b`.
    aR, bR: float
        Radiuses of the spheres `a` and `b`.

    Returns
    -------
    inside: boolean
        True if `b` is entirely inside `a`.
    '''
    return np.logical_not(spheres_intersect(ar, -aR, br, bR))


def cap_insphere_intersect(ar1, ar2, aR, br, bR):
    '''
    Returns True if a
    [capsule](http://en.wikipedia.org/wiki/Capsule_(geometry))
    is entirely inside a sphere.

    Parameters
    ----------
    ar1, ar2: array-like, shape (n,) in n dimensions
        Coordinates for the points of the line segment defining the capsule.
        The hemispheres are centred on these points.
    aR: float
        Radius of the capsule hemispheres, and cylinder sections.
    br: array-like, shape (n,)
        Coordinates of the centre of the sphere.
    bR: float
        Radius of the sphere

    Returns
    -------
    inside: boolean
        True if the capsule is entirely inside the sphere.
    '''
    return (np.logical_or(sphere_insphere_intersect(ar1, aR, br, bR),
                          sphere_insphere_intersect(ar2, aR, br, bR)))


def cap_insphere_sep(ar1, ar2, aR, br, bR):
    '''
    Returns the maximum distance of a
    [capsule](http://en.wikipedia.org/wiki/Capsule_(geometry)) from a sphere.

    Parameters
    ----------
    ar1, ar2: array-like, shape (n,) in n dimensions
        Coordinates for the points of the line segment defining the capsule.
        The hemispheres are centred on these points.
    aR: float
        Radius of the capsule hemispheres, and cylinder sections.
    br: array-like, shape (n,)
        Coordinates of the centre of the sphere.
    bR: float
        Radius of the sphere

    Returns
    -------
    d: float
        The maximum distance of the capsule from the sphere.
    '''
    ds = np.array([sphere_insphere_sep(ar1, aR, br, bR),
                   sphere_insphere_sep(ar2, aR, br, bR)])
    r = np.where((np.argmax(ds, axis=0))[:, np.newaxis], ar2, ar1)
    return r - br


def caps_intersect(ar1, ar2, aR, br1, br2, bR):
    '''
    Returns True if two
    [capsule](http://en.wikipedia.org/wiki/Capsule_(geometry)) `a` and `b`
    intersect each other, in 3 dimensions.

    Parameters
    ----------
    ar1, ar2, br1, br2: array-like, shape (3,) in 3 dimensions
        Coordinates for the points of the line segment defining capsules
        `a` and `b`.
        The hemispheres are centred on these points.
    aR, bR: float
        Radius of the capsule hemispheres, and cylinder sections.

    Returns
    -------
    intersect: boolean
        True if the capsules intersect each other
    '''
    return segs_sep_sq(ar1, ar2, br1, br2) < (aR + bR) ** 2


def caps_intersect_intro(r, u, l, R, L):
    '''
    For capsules in a periodic system, returns if each capsule intersects
    at least one other capsule.

    http://en.wikipedia.org/wiki/Capsule_(geometry)

    Parameters
    ----------
    r: array-like, shape (m, 3) for m capsules in 3 dimensions
        Coordinates for the centres of the capsules.
    u: array-like, shape (m, n)
        Unit vectors describing the orientation of the capsules.
        The vectors should point along the capsule's axis in either direction.
    l: float
        Length of the capsules' line segments.
    R: float
        Radius of the capsule hemispheres, and cylinder sections.
    L: float
        Length of the periodic system.

    Returns
    -------
    intersect: boolean array, shape (m,)
        True if a capsule intersects at least one other capsule.
    '''
    return geom_numerics.caps_intersect_intro(r, u, l, R, L)


def caps_sep_intro(r, u, l, R, L):
    '''
    For capsules in a periodic 3 dimensional system,
    returns the minimum separation vectors between each capsule and its
    nearest neighbour, if the capsule intersects its nearest neighbour.

    If a capsule doesn't intersect any other, the vector is not well-defined.

    http://en.wikipedia.org/wiki/Capsule_(geometry)

    Parameters
    ----------
    r: array-like, shape (m, 3) for m capsules in 3 dimensions
        Coordinates for the centres of the capsules.
    u: array-like, shape (m, 3)
        Unit vectors describing the orientation of the capsules.
        The vectors should point along the capsule's axis in either direction.
    l: float
        Length of the capsules' line segments.
    R: float
        Radius of the capsule hemispheres, and cylinder sections.
    L: float
        Length of the periodic system.

    Returns
    -------
    d: float array, shape (m, 3)
        Minimum separation vectors between each capsule and its
        nearest neighbour, if the two intersect each other.
    '''
    return geom_numerics.caps_sep_intro(r, u, l, R, L)


def point_seg_sep(ar, br1, br2):
    '''
    Returns the minimum separation vector between a point and a line segment,
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
    '''
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
    '''
    Returns the squared minimum separation distance between a point and a
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
    '''
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


def segs_sep_sq(ar1, ar2, br1, br2):
    '''
    Returns the squared minimum separation distance between two line segments,
    in 3 dimensions.

    Parameters
    ----------
    ar1, ar2, br1, br2: array-like, shape (3,)
        Coordinates for the points of two line segments `a` and `b`.

    Returns
    -------
    d_sq: float
        Squared separation distance between the two line segments.
    '''
    u = ar2 - ar1
    v = br2 - br1
    w = ar1 - br1

    a = np.sum(np.square(u))
    b = np.sum(u * v)
    c = np.sum(np.square(v))
    d = np.sum(u * w)
    e = np.sum(v * w)
    D = a * c - b ** 2
    sc = sN = sD = D
    tc = tN = tD = D

    if D < SMALL:
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = b * e - c * d
        tN = a * e - b * d
        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c

    if tN < 0.0:
        tN = 0.0

        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD

        if (-d + b) < 0.0:
            sN = 0.0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = -d + b
            sD = a

    sc = 0.0 if abs(sN) < SMALL else sN / sD
    tc = 0.0 if abs(tN) < SMALL else tN / tD

    sep = w + (sc * u) - (tc * v)
    return np.sum(np.square(sep))


def angular_distance(n1, n2):
    '''
    Returns the angular separation between two 3 dimensional vectors,
    when embedded onto the surface of the unit sphere centred at the origin.

    Parameters
    ----------
    n1, n2: array-like, shape (3,)
        Coordinates of two vectors.
        The magnitude of the vectors does not matter as both are normalised
        to unit vectors.

    Returns
    -------
    d_sigma: float
        Angle between unit vectors of n1 and n2 in radians.
    '''
    r1, r2 = utils.vector_mag(n1), utils.vector_mag(n2)
    u1, u2 = n1 / r1, n2 / r2
    return np.arctan2(utils.vector_mag(np.cross(u1, u2)), np.dot(u1, u2))
