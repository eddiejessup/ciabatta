import numpy as np
from ciabatta.cell_list import intro as cl_intro
from libcpp cimport bool
cimport numpy as np
cimport cython


cdef double SMALL = 1e-10


@cython.cdivision(True)
@cython.boundscheck(False)
cdef double line_segments_sep_sq(
        np.ndarray[np.float_t, ndim=1] s1,
        np.ndarray[np.float_t, ndim=1] s2,
        np.ndarray[np.float_t, ndim=1] w):
    '''
    Returns the squared minimum separation distance between two line segments,
    in 3 dimensions.

    For two line segments `a` and `b` with end-points ar1, ar2, br1, br2.

    Parameters
    ----------
    s1 = ar2 - ar1
    s2 = br2 - br1
    w = ar1 - br1

    Returns
    -------
    d_sq: float
        Squared separation distance between the two line segments.
    '''
    cdef:
        double a = 0.0, b = 0.0, c = 0.0, d = 0.0, e = 0.0
        double D, s1N, s1D, s1c, s2N, s2D, s2c, sep_sq = 0.0
        unsigned int dim = s1.shape[0], i

    for i in range(dim):
        a += s1[i] ** 2
        b += s1[i] * s2[i]
        c += s2[i] ** 2
        d += s1[i] * w[i]
        e += s2[i] * w[i]

    D = a * c - b ** 2

    s1c = s1N = s1D = D
    s2c = s2N = s2D = D

    if D < SMALL:
        s1N = 0.0
        s1D = 1.0
        s2N = e
        s2D = c
    else:
        s1N = b * e - c * d
        s2N = a * e - b * d
        if s1N < 0.0:
            s1N = 0.0
            s2N = e
            s2D = c
        elif s1N > s1D:
            s1N = s1D
            s2N = e + b
            s2D = c

    if s2N < 0.0:
        s2N = 0.0

        if -d < 0.0:
            s1N = 0.0
        elif -d > a:
            s1N = s1D
        else:
            s1N = -d
            s1D = a
    elif s2N > s2D:
        s2N = s2D

        if (-d + b) < 0.0:
            s1N = 0.0
        elif (-d + b ) > a:
            s1N = s1D
        else:
            s1N = -d + b
            s1D = a

    if abs(s1N) < SMALL:
        s1c = 0.0
    else:
        s1c = s1N / s1D
    if abs(s2N) < SMALL:
        s2c = 0.0
    else:
        s2c = s2N / s2D

    for i in range(dim):
        sep_sq += (w[i] + (s1c * s1[i]) - (s2c * s2[i])) ** 2

    return sep_sq


@cython.cdivision(True)
@cython.boundscheck(False)
def capsule_intersection(np.ndarray[np.float_t, ndim=2] r,
                         np.ndarray[np.float_t, ndim=2] u,
                         double l, double R, double L):
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
    cdef:
        unsigned int i, i_i2, i2, idim, n = r.shape[0], dim = r.shape[1]
        np.ndarray[np.uint8_t, ndim=1, cast=True] collisions = np.zeros((n,), dtype=np.bool)
        np.ndarray[int, ndim=2] inters
        np.ndarray[int, ndim=1] intersi
        tuple dims = (dim,)
        np.ndarray[np.float_t, ndim=1] s1 = np.zeros(dims), s2 = np.zeros(dims), wd = np.zeros(dims), r1d = np.zeros(dims)
        double sep_sq_max = (2.0 * R) ** 2, l_half = l / 2.0

    inters, intersi = cl_intro.get_inters(r, L, 2.0 * R + l)

    for i in range(n):
        if intersi[i] > 0:
            for idim in range(dim):
                s1[idim] = u[i, idim] * l
                r1d[idim] = r[i, idim] - u[i, idim] * l_half
        for i_i2 in range(intersi[i]):
            i2 = inters[i, i_i2]
            for idim in range(dim):
                s2[idim] = u[i2, idim] * l
                wd[idim] = r1d[idim] - (r[i2, idim] - u[i2, idim] * l_half)
            if line_segments_sep_sq(s1, s2, wd) < sep_sq_max:
                collisions[i] = True
                break

    return collisions


@cython.cdivision(True)
@cython.boundscheck(False)
def sphere_intersection(np.ndarray[np.float_t, ndim=2] r,
                        float R):
    cdef:
        unsigned int i, j
        np.ndarray[np.uint8_t, ndim=1] c = np.zeros(r.shape[0], dtype=np.uint8)

    for i in range(r.shape[0]):
        for j in range(r.shape[0]):
            if i != j and ((r[i, 0] - r[j, 0]) ** 2 +
                           (r[i, 1] - r[j, 1]) ** 2 +
                           (r[i, 2] - r[j, 2]) ** 2 < (2.0 * R) ** 2):
                c[i] = 1
                break
    return np.array(c, dtype=np.bool)
