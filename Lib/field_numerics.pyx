import numpy as np
import utils
cimport numpy as np

cdef unsigned int wrap_inc(unsigned int M, unsigned int i):
    return i + 1 if i < M - 1 else 0

cdef unsigned int wrap_dec(unsigned int M, unsigned int i):
    return i - 1 if i > 0 else M - 1

def div(field, dx):
    assert dx > 0.0
    div = np.empty(field.shape[:-1], dtype=field.dtype)    
    if field.ndim == 2: div_1d(field, div, dx)
    elif field.ndim == 3: div_2d(field, div, dx)
    elif field.ndim == 4: div_3d(field, div, dx)
    else: raise Exception('Divergence not implemented in this dimension')
    return div

def div_1d(np.ndarray[np.float_t, ndim=2] field, 
        np.ndarray[np.float_t, ndim=1] div,
        double dx):
    cdef unsigned int i_x
    cdef unsigned int M_x = field.shape[0]
    cdef double dx_double = 2.0 * dx

    for i_x in range(M_x):
        div[i_x] = (field[wrap_inc(M_x, i_x), 0] - field[wrap_dec(M_x, i_x), 0]) / dx_double

def div_2d(np.ndarray[np.float_t, ndim=3] field, 
        np.ndarray[np.float_t, ndim=2] div,
        double dx):
    cdef unsigned int i_x, i_y
    cdef unsigned int M_x = field.shape[0], M_y = field.shape[1]
    cdef double dx_double = 2.0 * dx

    for i_x in range(M_x):
        for i_y in range(M_y):
            div[i_x, i_y] = (
                (field[wrap_inc(M_x, i_x), i_y, 0] - field[wrap_dec(M_x, i_x), i_y, 0]) + 
                (field[i_x, wrap_inc(M_y, i_y), 1] - field[i_x, wrap_dec(M_y, i_y), 1]) + 
                (field[wrap_inc(M_x, i_x), i_y, 0] - field[wrap_dec(M_x, i_x), i_y, 0])) / dx_double

def div_3d(np.ndarray[np.float_t, ndim=4] field, 
        np.ndarray[np.float_t, ndim=3] div,
        double dx):
    cdef unsigned int i_x, i_y, i_z
    cdef unsigned int M_x = field.shape[0], M_y = field.shape[1]
    cdef unsigned int M_z = field.shape[2]
    cdef double dx_double = 2.0 * dx

    for i_x in range(M_x):
        for i_y in range(M_y):
            for i_z in range(M_z):
                div[i_x, i_y, i_z] = (
                    (field[wrap_inc(M_x, i_x), i_y, i_z, 0] - field[wrap_dec(M_x, i_x), i_y, i_z, 0]) + 
                    (field[i_x, wrap_inc(M_y, i_y), i_z, 1] - field[i_x, wrap_dec(M_y, i_y), i_z, 1]) + 
                    (field[i_x, i_y, wrap_inc(M_z, i_z), 2] - field[i_x, i_y, wrap_dec(M_z, i_z), 2])) / dx_double

def grad(field, dx):
    assert dx > 0.0
    grad = np.empty(field.shape + (field.ndim,), dtype=field.dtype)
    if field.ndim == 1: grad_1d(field, grad, dx)
    elif field.ndim == 2: grad_2d(field, grad, dx)
    elif field.ndim == 3: grad_3d(field, grad, dx)
    else: raise Exception('Grad not implemented in this dimension')
    return grad

def grad_1d(np.ndarray[np.float_t, ndim=1] field, 
        np.ndarray[np.float_t, ndim=2] grad,
        double dx):
    cdef unsigned int i_x
    cdef unsigned int M_x = field.shape[0]
    cdef double dx_double = 2.0 * dx

    for i_x in range(M_x):
        grad[i_x, 0] = (field[wrap_inc(M_x, i_x)] - field[wrap_dec(M_x, i_x)]) / dx_double

def grad_2d(np.ndarray[np.float_t, ndim=2] field, 
        np.ndarray[np.float_t, ndim=3] grad,
        double dx):
    cdef unsigned int i_x, i_y
    cdef unsigned int M_x = field.shape[0], M_y = field.shape[1]
    cdef double dx_double = 2.0 * dx

    for i_x in range(M_x):
        for i_y in range(M_y):
            grad[i_x, i_y, 0] = (field[wrap_inc(M_x, i_x), i_y] - field[wrap_dec(M_x, i_x), i_y]) / dx_double
            grad[i_x, i_y, 1] = (field[i_x, wrap_inc(M_y, i_y)] - field[i_x, wrap_dec(M_y, i_y)]) / dx_double

def grad_3d(np.ndarray[np.float_t, ndim=3] field, 
        np.ndarray[np.float_t, ndim=4] grad,
        double dx):
    cdef unsigned int i_x, i_y, i_z
    cdef unsigned int M_x = field.shape[0], M_y = field.shape[1]
    cdef unsigned int M_z = field.shape[2]
    cdef double dx_double = 2.0 * dx

    for i_x in range(M_x):
        for i_y in range(M_y):
            for i_z in range(M_z):
                grad[i_x, i_y, i_z, 0] = (field[wrap_inc(M_x, i_x), i_y, i_z] - field[wrap_dec(M_x, i_x), i_y, i_z]) / dx_double
                grad[i_x, i_y, i_z, 1] = (field[i_x, wrap_inc(M_y, i_y), i_z] - field[i_x, wrap_dec(M_y, i_y), i_z]) / dx_double
                grad[i_x, i_y, i_z, 2] = (field[i_x, i_y, wrap_inc(M_z, i_z)] - field[i_x, i_y, wrap_dec(M_z, i_z)]) / dx_double

def grad_i(field, inds, dx):
    assert dx > 0.0
    assert inds.ndim == 2
    assert field.ndim == inds.shape[1]
    grad_i = np.empty(inds.shape, dtype=field.dtype)
    if field.ndim == 1: grad_i_1d(field, inds, grad_i, dx)
    elif field.ndim == 2: grad_i_2d(field, inds, grad_i, dx)
    elif field.ndim == 3: grad_i_3d(field, grad_i, dx)
    else: raise Exception("Grad_i not implemented in this dimension")
    return grad_i

def grad_i_1d(np.ndarray[np.float_t, ndim=1] field, 
        np.ndarray[np.int_t, ndim=2] inds,
        np.ndarray[np.float_t, ndim=2] grad_i,
        double dx):
    cdef unsigned int i, i_x
    cdef unsigned int M_x = field.shape[0]
    cdef double dx_double = 2.0 * dx

    for i in range(inds.shape[0]):
        i_x = inds[i, 0]
        grad_i[i, 0] = (field[wrap_inc(M_x, i_x)] - field[wrap_dec(M_x, i_x)]) / dx_double

def grad_i_2d(np.ndarray[np.float_t, ndim=2] field, 
        np.ndarray[np.int_t, ndim=2] inds,
        np.ndarray[np.float_t, ndim=2] grad_i,
        double dx):
    cdef unsigned int i, i_x, i_y
    cdef unsigned int M_x = field.shape[0], M_y = field.shape[1]
    cdef double dx_double = 2.0 * dx

    for i in range(inds.shape[0]):
        i_x, i_y = inds[i, 0], inds[i, 1]
        grad_i[i, 0] = (field[wrap_inc(M_x, i_x), i_y] - field[wrap_dec(M_x, i_x), i_y]) / dx_double
        grad_i[i, 1] = (field[i_x, wrap_inc(M_y, i_y)] - field[i_x, wrap_dec(M_y, i_y)]) / dx_double

def grad_i_3d(np.ndarray[np.float_t, ndim=3] field, 
        np.ndarray[np.int_t, ndim=2] inds,
        np.ndarray[np.float_t, ndim=2] grad_i,
        double dx):
    cdef unsigned int i, i_x, i_y, i_z
    cdef unsigned int M_x = field.shape[0], M_y = field.shape[1]
    cdef unsigned int M_z = field.shape[2]
    cdef double dx_double = 2.0 * dx

    for i in range(inds.shape[0]):
        i_x, i_y, i_z = inds[i, 0], inds[i, 1], inds[i, 2]
        grad_i[i, 0] = (field[wrap_inc(M_x, i_x), i_y, i_z] - field[wrap_dec(M_x, i_x), i_y, i_z]) / dx_double
        grad_i[i, 1] = (field[i_x, wrap_inc(M_y, i_y), i_z] - field[i_x, wrap_dec(M_y, i_y), i_z]) / dx_double
        grad_i[i, 2] = (field[i_x, i_y, wrap_inc(M_z, i_z)] - field[i_x, i_y, wrap_dec(M_z, i_z)]) / dx_double

def laplace(field, dx):
    assert dx > 0.0
    laplace = np.empty_like(field)
    if field.ndim == 1: laplace_1d(field, laplace, dx)
    elif field.ndim == 2: laplace_2d(field, laplace, dx)
    elif field.ndim == 3: laplace_3d(field, laplace, dx)
    else: raise Exception('Laplacian not implemented in this dimension')
    return laplace 

def laplace_1d(np.ndarray[np.float_t, ndim=1] field, 
        np.ndarray[np.float_t, ndim=1] laplace,
        double dx):
    cdef unsigned int i_x
    cdef unsigned int M_x = field.shape[0]
    cdef double dx_sq = dx * dx

    for i_x in range(M_x):
        laplace[i_x] = (
            field[wrap_inc(M_x, i_x)] + field[wrap_dec(M_x, i_x)] - 
            2.0 * field[i_x]) / dx_sq

def laplace_2d(np.ndarray[np.float_t, ndim=2] field, 
        np.ndarray[np.float_t, ndim=2] laplace,
        double dx):
    cdef unsigned int i_x, i_y
    cdef unsigned int M_x = field.shape[0], M_y = field.shape[1]
    cdef double dx_sq = dx * dx, diff

    for i_x in range(M_x):
        for i_y in range(M_y):
            laplace[i_x, i_y] = (
                field[wrap_inc(M_x, i_x), i_y] + field[wrap_dec(M_x, i_x), i_y] + 
                field[i_x, wrap_inc(M_y, i_y)] + field[i_x, wrap_dec(M_y, i_y)] - 
                4.0 * field[i_x, i_y]) / dx_sq

def laplace_3d(np.ndarray[np.float_t, ndim=3] field, 
        np.ndarray[np.float_t, ndim=3] laplace,
        double dx):
    cdef unsigned int i_x, i_y, i_z
    cdef unsigned int M_x = field.shape[0], M_y = field.shape[1]
    cdef unsigned int M_z = field.shape[2]
    cdef double dx_sq = dx * dx

    for i_x in range(M_x):
        for i_y in range(M_y):
            for i_z in range(M_z):
                laplace[i_x, i_y, i_z] = (
                    field[wrap_inc(M_x, i_x), i_y, i_z] + field[wrap_dec(M_x, i_x), i_y, i_z] +
                    field[i_x, wrap_inc(M_y, i_y), i_z] + field[i_x, wrap_dec(M_y, i_y), i_z] + 
                    field[i_x, i_y, wrap_inc(M_z, i_z)] + field[i_x, i_y, wrap_dec(M_z, i_z)] - 
                    6.0 * field[i_x, i_y, i_z]) / dx_sq

def density(r, L, dx):
    assert r.ndim == 2
    if (L / dx) % 1 != 0:
        raise Exception
    M = int(L / dx)
    inds = utils.r_to_i(r, L, dx)
    f = np.zeros(r.shape[1] * (M,), dtype=np.int)
    if f.ndim == 1: density_1d(inds, f)
    elif f.ndim == 2: density_2d(inds, f)
    elif f.ndim == 3: density_3d(inds, f)
    else: raise Exception('Density calc not implemented in this dimension')
    return f / dx ** r.shape[1]

def density_1d(np.ndarray[np.int_t, ndim=2] inds, 
        np.ndarray[np.int_t, ndim=1] f):
    cdef unsigned int i_part
    for i_part in range(inds.shape[0]):
        f[inds[i_part, 0]] += 1

def density_2d(np.ndarray[np.int_t, ndim=2] inds, 
        np.ndarray[np.int_t, ndim=2] f):
    cdef unsigned int i_part
    for i_part in range(inds.shape[0]):
        f[inds[i_part, 0], inds[i_part, 1]] += 1

def density_3d(np.ndarray[np.int_t, ndim=2] inds, 
        np.ndarray[np.int_t, ndim=3] f):
    cdef unsigned int i_part
    for i_part in range(inds.shape[0]):
        f[inds[i_part, 0], inds[i_part, 1], inds[i_part, 2]] += 1
