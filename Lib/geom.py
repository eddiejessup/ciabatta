import numpy as np
import utils
import geom_numerics

SMALL = 1e-10

def sphere_volume(R, n):
    '''
    Volume of an n-dimensional sphere of radius R.
    '''
    return ((np.pi ** (n / 2.0)) / scipy.special.gamma(n / 2.0 + 1)) * R ** n

def sphere_radius(V, n):
    '''
    Radius of an n-dimensional sphere with volume V.
    '''
    return ((scipy.special.gamma(n / 2.0 + 1.0) * V) ** (1.0 / n)) / np.sqrt(np.pi)

def sphere_area(R, n):
    '''
    Surface area of an n-dimensional sphere of radius R.
    NOTE: In 2d this will return a circle's circumference, not what is commonly
    referred to as its area.
    '''
    return (n / R) * sphere_volume(R, n)

def cylinder_volume(R, l):
    '''
    Volume of a cylinder with radius R and length l.
    '''
    # Remember the volume of a sphere in 2d is what's usually referred to as its area
    return sphere_volume(R, 2) * l

def cylinder_area(R, l):
    '''
    Surface area of a cylinder with radius R and length l.
    '''
    return sphere_area(R, 2) * l

def capsule_volume(R, l):
    '''
    Volume of a capsule with radius R and line segment length l.
    '''
    return sphere_volume(R, 3) + cylinder_volume(R, l)

def capsule_area(R, l):
    '''
    Surface area of a capsule with radius R and line segment length l.
    '''
    return sphere_area(R, 3) + cylinder_area(R, l)

def spheres_sep(ar, aR, br, bR):
    '''
    For two spheres centred at ar; br, with radii aR; bR,
    the separation distance between the two (negative means intersecting).
    '''
    return utils.vector_mag(ar - br) - (aR + bR)

def spheres_intersect(ar, aR, br, bR):
    '''
    For two spheres centred at ar; br, with radii aR; bR,
    True if they intersect.
    '''
    return utils.vector_mag_sq(ar - br) < (aR + bR) ** 2

def sphere_insphere_sep(ar, aR, br, bR):
    '''
    For two spheres centred at ar; br, with radii aR; bR,
    the amount by which sphere b is inside sphere a (negative means outside).
    '''
    return spheres_sep(ar, -aR, br, bR)

def sphere_insphere_intersect(ar, aR, br, bR):
    '''
    For two spheres centred at ar; br, with radii aR; bR,
    True if sphere a is fully contained by sphere b.
    '''
    return np.logical_not(spheres_intersect(ar, -aR, br, bR))

def cap_insphere_intersect(ar1, ar2, aR, br, bR):
    '''
    For a capsule defined by the line segment (ar1, ar2) with radius aR,
    and a sphere centred at br with radius bR, 
    True if the capsule is fully contained within the sphere.
    '''
    return np.logical_or(sphere_insphere_intersect(ar1, aR, br, bR), sphere_insphere_intersect(ar2, aR, br, bR))

def cap_insphere_sep(ar1, ar2, aR, br, bR):
    '''
    For a capsule with ends ar1, ar2, radius aR;
    inside a sphere at br, radius bR;
    the maximum distance of the capsule from br.
    '''
    ds = np.array([sphere_insphere_sep(ar1, aR, br, bR), sphere_insphere_sep(ar2, aR, br, bR)])
    r = np.where((np.argmax(ds, axis=0))[:, np.newaxis], ar2, ar1)
    return r - br

def caps_intersect(ar1, ar2, aR, br1, br2, bR):
    '''
    For two capsules defined by line segments (ar1, ar2) and (br1, br2),
    with radii aR and bR, 
    True if they intersect.
    '''
    return segs_sep_sq(ar1, ar2, br1, br2) < (aR + bR) ** 2

def caps_intersect_intro(r, u, lu, ld, R, L):
    '''
    For capsules with centres r, orientation u, forward length lu,
    backward length ld, radius R in a periodic system of period L,
    a boolean array representing if each capsule intersects
    at least one other capsule.
    '''
    return geom_numerics.caps_intersect_intro(r, u, lu, ld, R, L)

def caps_sep_intro(r, u, lu, ld, R, L):
    '''
    For capsules with centres r, orientation u, forward length lu,
    backward length ld, radius R in a periodic system of period L,
    the minimum separation vectors between each capsule and its
    closest neighbour.
    NOTE: the vector is ONLY well-defined if that neighbour intersects the capsule.
    '''
    return geom_numerics.caps_sep_intro(r, u, lu, ld, R, L)

def point_seg_sep_sq(ar, br1, br2):
    '''
    For a point at ar, and a line segment between br1 and br2,
    the square of the minimum distance between them.
    '''
    v = br2 - br1
    w = ar - br1

    c1 = np.dot(w, v)
    if c1 <= 0.0:
        return np.sum(np.square(ar - br1))

    c2 = np.sum(np.square(v));
    if c2 <= c1:
        return np.sum(np.square(ar - br2))

    b = c1 / c2
    bc = br1 + b * v
    return np.sum(np.square(ar - bc))

def segs_sep_sq(ar1, ar2, br1, br2):
    '''
    For two line segments between points (ar1, ar2) and (br1, br2),
    the square of the minimum distance between them.
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
        elif (-d + b ) > a:
            sN = sD
        else:
            sN = -d + b
            sD = a

    sc = 0.0 if abs(sN) < SMALL else sN / sD
    tc = 0.0 if abs(tN) < SMALL else tN / tD

    sep = w + (sc * u) - (tc * v)
    return np.sum(np.square(sep))
