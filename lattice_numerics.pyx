import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport int


@cython.cdivision(True)
@cython.boundscheck(False)
def r_to_i(np.ndarray[np.float_t, ndim=2] r,
           double L, double dx):
    cdef:
        unsigned int i, idim
        double L_half = L / 2.0
        np.ndarray[np.int_t, ndim=2] ind = np.empty([r.shape[0], r.shape[1]], dtype=np.int)

    for idim in range(r.shape[1]):
        for i in range(r.shape[0]):
            ind[i, idim] = int((r[i, idim] + L_half) / dx)
    return ind
