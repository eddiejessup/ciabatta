cimport cython

cdef inline double square(double f):
    return f * f

cdef inline double mag_sq(double x, double y):
    return square(x) + square(y)

cdef inline double wrap_real(double L, double L_half, double r):
    if r > L_half: return r - L
    elif r < -L_half: return r + L
    else: return r

cdef inline unsigned int wrap(unsigned int M, int i):
    if i >= M: return i - M
    elif i < 0: return i + M
    else: return i

cdef inline unsigned int wrap_inc(unsigned int M, unsigned int i):
    return i + 1 if i < M - 1 else 0

cdef inline unsigned int wrap_dec(unsigned int M, unsigned int i):
    return i - 1 if i > 0 else M - 1
