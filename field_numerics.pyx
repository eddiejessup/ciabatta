import numpy as np
cimport numpy as np
cimport cython


cdef unsigned int wrap_inc(unsigned int M, unsigned int i):
    return i + 1 if i < M - 1 else 0


cdef unsigned int wrap_dec(unsigned int M, unsigned int i):
    return i - 1 if i > 0 else M - 1


@cython.cdivision(True)
@cython.boundscheck(False)
def div_1d(np.ndarray[np.float_t, ndim=2] field,
           np.ndarray[np.float_t, ndim=1] div,
           double dx):
    cdef:
        unsigned int i_x
        unsigned int M_x = field.shape[0]
        double dx_double = 2.0 * dx

    for i_x in range(M_x):
        div[i_x] = (field[wrap_inc(M_x, i_x), 0] - field[wrap_dec(M_x, i_x), 0]) / dx_double


@cython.cdivision(True)
@cython.boundscheck(False)
def div_2d(np.ndarray[np.float_t, ndim=3] field,
           np.ndarray[np.float_t, ndim=2] div,
           double dx):
    cdef:
        unsigned int i_x, i_y
        unsigned int M_x = field.shape[0], M_y = field.shape[1]
        double dx_double = 2.0 * dx

    for i_x in range(M_x):
        for i_y in range(M_y):
            div[i_x, i_y] = (
                (field[wrap_inc(M_x, i_x), i_y, 0] - field[wrap_dec(M_x, i_x), i_y, 0]) +
                (field[i_x, wrap_inc(M_y, i_y), 1] - field[i_x, wrap_dec(M_y, i_y), 1]) +
                (field[wrap_inc(M_x, i_x), i_y, 0] - field[wrap_dec(M_x, i_x), i_y, 0])) / dx_double


@cython.cdivision(True)
@cython.boundscheck(False)
def div_3d(np.ndarray[np.float_t, ndim=4] field,
           np.ndarray[np.float_t, ndim=3] div,
           double dx):
    cdef:
        unsigned int i_x, i_y, i_z
        unsigned int M_x = field.shape[0], M_y = field.shape[1]
        unsigned int M_z = field.shape[2]
        double dx_double = 2.0 * dx

    for i_x in range(M_x):
        for i_y in range(M_y):
            for i_z in range(M_z):
                div[i_x, i_y, i_z] = (
                    (field[wrap_inc(M_x, i_x), i_y, i_z, 0] - field[wrap_dec(M_x, i_x), i_y, i_z, 0]) +
                    (field[i_x, wrap_inc(M_y, i_y), i_z, 1] - field[i_x, wrap_dec(M_y, i_y), i_z, 1]) +
                    (field[i_x, i_y, wrap_inc(M_z, i_z), 2] - field[i_x, i_y, wrap_dec(M_z, i_z), 2])) / dx_double


@cython.cdivision(True)
@cython.boundscheck(False)
def grad_1d(np.ndarray[np.float_t, ndim=1] field,
            np.ndarray[np.float_t, ndim=2] grad,
            double dx):
    cdef:
        unsigned int i_x
        unsigned int M_x = field.shape[0]
        double dx_double = 2.0 * dx

    for i_x in range(M_x):
        grad[i_x, 0] = (field[wrap_inc(M_x, i_x)] - field[wrap_dec(M_x, i_x)]) / dx_double


@cython.cdivision(True)
@cython.boundscheck(False)
def grad_2d(np.ndarray[np.float_t, ndim=2] field,
            np.ndarray[np.float_t, ndim=3] grad,
            double dx):
    cdef:
        unsigned int i_x, i_y
        unsigned int M_x = field.shape[0], M_y = field.shape[1]
        double dx_double = 2.0 * dx

    for i_x in range(M_x):
        for i_y in range(M_y):
            grad[i_x, i_y, 0] = (field[wrap_inc(M_x, i_x), i_y] - field[wrap_dec(M_x, i_x), i_y]) / dx_double
            grad[i_x, i_y, 1] = (field[i_x, wrap_inc(M_y, i_y)] - field[i_x, wrap_dec(M_y, i_y)]) / dx_double


@cython.cdivision(True)
@cython.boundscheck(False)
def grad_3d(np.ndarray[np.float_t, ndim=3] field,
            np.ndarray[np.float_t, ndim=4] grad,
            double dx):
    cdef:
        unsigned int i_x, i_y, i_z
        unsigned int M_x = field.shape[0], M_y = field.shape[1]
        unsigned int M_z = field.shape[2]
        double dx_double = 2.0 * dx

    for i_x in range(M_x):
        for i_y in range(M_y):
            for i_z in range(M_z):
                grad[i_x, i_y, i_z, 0] = (field[wrap_inc(M_x, i_x), i_y, i_z] - field[wrap_dec(M_x, i_x), i_y, i_z]) / dx_double
                grad[i_x, i_y, i_z, 1] = (field[i_x, wrap_inc(M_y, i_y), i_z] - field[i_x, wrap_dec(M_y, i_y), i_z]) / dx_double
                grad[i_x, i_y, i_z, 2] = (field[i_x, i_y, wrap_inc(M_z, i_z)] - field[i_x, i_y, wrap_dec(M_z, i_z)]) / dx_double


@cython.cdivision(True)
@cython.boundscheck(False)
def grad_i_1d(np.ndarray[np.float_t, ndim=1] field,
              np.ndarray[np.int_t, ndim=2] inds,
              np.ndarray[np.float_t, ndim=2] grad_i,
              double dx):
    cdef:
        unsigned int i, i_x
        unsigned int M_x = field.shape[0]
        double dx_double = 2.0 * dx

    for i in range(inds.shape[0]):
        i_x = inds[i, 0]
        grad_i[i, 0] = (field[wrap_inc(M_x, i_x)] - field[wrap_dec(M_x, i_x)]) / dx_double


@cython.cdivision(True)
@cython.boundscheck(False)
def grad_i_2d(np.ndarray[np.float_t, ndim=2] field,
              np.ndarray[np.int_t, ndim=2] inds,
              np.ndarray[np.float_t, ndim=2] grad_i,
              double dx):
    cdef:
        unsigned int i, i_x, i_y
        unsigned int M_x = field.shape[0], M_y = field.shape[1]
        double dx_double = 2.0 * dx

    for i in range(inds.shape[0]):
        i_x, i_y = inds[i, 0], inds[i, 1]
        grad_i[i, 0] = (field[wrap_inc(M_x, i_x), i_y] - field[wrap_dec(M_x, i_x), i_y]) / dx_double
        grad_i[i, 1] = (field[i_x, wrap_inc(M_y, i_y)] - field[i_x, wrap_dec(M_y, i_y)]) / dx_double


@cython.cdivision(True)
@cython.boundscheck(False)
def grad_i_3d(np.ndarray[np.float_t, ndim=3] field,
              np.ndarray[np.int_t, ndim=2] inds,
              np.ndarray[np.float_t, ndim=2] grad_i,
              double dx):
    cdef:
        unsigned int i, i_x, i_y, i_z
        unsigned int M_x = field.shape[0], M_y = field.shape[1]
        unsigned int M_z = field.shape[2]
        double dx_double = 2.0 * dx

    for i in range(inds.shape[0]):
        i_x, i_y, i_z = inds[i, 0], inds[i, 1], inds[i, 2]
        grad_i[i, 0] = (field[wrap_inc(M_x, i_x), i_y, i_z] - field[wrap_dec(M_x, i_x), i_y, i_z]) / dx_double
        grad_i[i, 1] = (field[i_x, wrap_inc(M_y, i_y), i_z] - field[i_x, wrap_dec(M_y, i_y), i_z]) / dx_double
        grad_i[i, 2] = (field[i_x, i_y, wrap_inc(M_z, i_z)] - field[i_x, i_y, wrap_dec(M_z, i_z)]) / dx_double


@cython.cdivision(True)
@cython.boundscheck(False)
def laplace_1d(np.ndarray[np.float_t, ndim=1] field,
               np.ndarray[np.float_t, ndim=1] laplace,
               double dx):
    cdef:
        unsigned int i_x
        unsigned int M_x = field.shape[0]
        double dx_sq = dx * dx

    for i_x in range(M_x):
        laplace[i_x] = (
            field[wrap_inc(M_x, i_x)] + field[wrap_dec(M_x, i_x)] -
            2.0 * field[i_x]) / dx_sq


def laplace_2d(np.ndarray[np.float_t, ndim=2] field,
               np.ndarray[np.float_t, ndim=2] laplace,
               double dx):
    cdef:
        unsigned int i_x, i_y
        unsigned int M_x = field.shape[0], M_y = field.shape[1]
        double dx_sq = dx * dx, diff

    for i_x in range(M_x):
        for i_y in range(M_y):
            laplace[i_x, i_y] = (
                field[wrap_inc(M_x, i_x), i_y] + field[wrap_dec(M_x, i_x), i_y] +
                field[i_x, wrap_inc(M_y, i_y)] + field[i_x, wrap_dec(M_y, i_y)] -
                4.0 * field[i_x, i_y]) / dx_sq


@cython.cdivision(True)
@cython.boundscheck(False)
def laplace_3d(np.ndarray[np.float_t, ndim=3] field,
               np.ndarray[np.float_t, ndim=3] laplace,
               double dx):
    cdef:
        unsigned int i_x, i_y, i_z
        unsigned int M_x = field.shape[0], M_y = field.shape[1]
        unsigned int M_z = field.shape[2]
        double dx_sq = dx * dx

    for i_x in range(M_x):
        for i_y in range(M_y):
            for i_z in range(M_z):
                laplace[i_x, i_y, i_z] = (
                    field[wrap_inc(M_x, i_x), i_y, i_z] + field[wrap_dec(M_x, i_x), i_y, i_z] +
                    field[i_x, wrap_inc(M_y, i_y), i_z] + field[i_x, wrap_dec(M_y, i_y), i_z] +
                    field[i_x, i_y, wrap_inc(M_z, i_z)] + field[i_x, i_y, wrap_dec(M_z, i_z)] -
                    6.0 * field[i_x, i_y, i_z]) / dx_sq


@cython.cdivision(True)
@cython.boundscheck(False)
def density_1d(np.ndarray[np.int_t, ndim=2] inds,
               np.ndarray[np.int_t, ndim=1] f):
    cdef:
        unsigned int i_part
    for i_part in range(inds.shape[0]):
        f[inds[i_part, 0]] += 1


@cython.cdivision(True)
@cython.boundscheck(False)
def density_2d(np.ndarray[np.int_t, ndim=2] inds,
               np.ndarray[np.int_t, ndim=2] f):
    cdef:
        unsigned int i_part
    for i_part in range(inds.shape[0]):
        f[inds[i_part, 0], inds[i_part, 1]] += 1


@cython.cdivision(True)
@cython.boundscheck(False)
def density_3d(np.ndarray[np.int_t, ndim=2] inds,
               np.ndarray[np.int_t, ndim=3] f):
    cdef:
        unsigned int i_part
    for i_part in range(inds.shape[0]):
        f[inds[i_part, 0], inds[i_part, 1], inds[i_part, 2]] += 1
