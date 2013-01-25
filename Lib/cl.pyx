import numpy as np
cimport numpy as np
import utils
from utils import wrap_inc, wrap_dec

filler = np.frompyfunc(lambda x: list(), 1, 1)

def get_cl(r, R, L):
    if r.shape[-1] == 2: return get_cl_2d(r, R, L)
    else: raise Exception('Dimension not supported for cell list')

def get_cl_2d(r, R, L):
    M = int(L / R.max())
    print M
    dx = L / M
    cl = np.empty([M, M], dtype=np.object)
    filler(cl, cl)
    inds = utils.r_to_i(r, L, dx)
    for i in range(len(inds)):
        cl[tuple(inds[i])].append(i)
    return cl

def get_checks(cl, r, L):
    if r.shape[-1] == 2: return get_checks_2d(cl, r, L)
    else: raise Exception('Dimension not supported for cell list')

def get_checks_2d(cl, r, L):
    M = cl.shape[0]
    dx = L / M
    inds = utils.r_to_i(r, L, dx)
    checks = np.empty([len(r)], dtype=np.object)
    for i in range(len(r)):
        x, y = inds[i]
        x_inc, y_inc = wrap_inc(M, x), wrap_inc(M, y)
        x_dec, y_dec = wrap_dec(M, x), wrap_dec(M, y)
        checks[i] = [
            (x, y),
            (x_inc, y), (x_dec, y), (x, y_inc), (x, y_dec),
            (x_inc, y_inc), (x_inc, y_dec), (x_dec, y_inc), (x_dec, y_dec)]
    return checks

def get_inters(checks, r, r_cl, R_cl, cl):
    if r.shape[-1] == 2: return get_inters_2d(checks, r, r_cl, R_cl, cl)
    else: raise Exception('Dimension not supported for cell list')

def get_inters_2d(np.ndarray checks,
        np.ndarray[np.float_t, ndim=2] r,
        np.ndarray[np.float_t, ndim=2] r_cl,
        np.ndarray[np.float_t, ndim=1] R_cl,
        np.ndarray cl):
    cdef unsigned int i, i_cl
    cdef tuple ind_cl
    cdef np.ndarray[np.float_t, ndim=1] R_cl_sq = R_cl ** 2
    cdef np.ndarray inters = np.empty([len(r)], dtype=np.object)
    filler(inters, inters)

    for i in range(len(r)):
        for ind_cl in checks[i]:
            for i_cl in cl[ind_cl[0], ind_cl[1]]:
                if (r[i, 0] - r_cl[i_cl, 0]) ** 2 + (r[i, 1] - r_cl[i_cl, 1]) ** 2 < R_cl_sq[i_cl]:
                    inters[i].append(i_cl)
    return inters