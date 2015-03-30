import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, atan2


@cython.cdivision(True)
@cython.boundscheck(False)
def pdist_angle(np.ndarray[np.float_t, ndim=2] r):
    '''
    Returns the absolute angular separation between many 3 dimensional vectors,
    when embedded onto the surface of the unit sphere centred at the origin.

    Parameters
    ----------
    r: array-like, shape (n, 3)
        Coordinates of `n` vectors. Their magnitude does not matter.

    Returns
    -------
    dsigmas: array, shape (n * (n - 1) / 2,)
        Separating angles in radians.
    '''
    cdef:
        unsigned int i1, i2
        double cross_mag, dot
        list seps = []

    for i1 in range(r.shape[0]):
        for i2 in range(i1 + 1, r.shape[0]):
            cross_mag = sqrt((r[i1, 1] * r[i2, 2] - r[i1, 2] * r[i2, 1]) ** 2 +
                             (r[i1, 2] * r[i2, 0] - r[i1, 0] * r[i2, 2]) ** 2 +
                             (r[i1, 0] * r[i2, 1] - r[i1, 1] * r[i2, 0]) ** 2)

            dot = r[i1, 0] * r[i2, 0] + r[i1, 1] * r[i2, 1] + r[i1, 2] * r[i2, 2]

            if cross_mag == 0.0 and dot == 0.0:
                seps.append(np.nan)
            else:
                seps.append(atan2(cross_mag, dot))
    return np.array(seps)
