import numpy as np
import utils
from utils import wrap_inc, wrap_dec

def get_cl(r, R, L):
    filler = np.frompyfunc(lambda x: list(), 1, 1)
    M = int(L / R.max())
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
    checks = [[] for i in range(len(r))]
    for i in range(len(r)):
        i_x, i_y = inds[i]
        checks[i] = cl[i_x, i_y][:]

        checks[i] += cl[wrap_inc(M, i_x), i_y][:]
        checks[i] += cl[wrap_dec(M, i_x), i_y][:]
        checks[i] += cl[i_x, wrap_inc(M, i_y)][:]
        checks[i] += cl[i_x, wrap_dec(M, i_y)][:]

        checks[i] += cl[wrap_inc(M, i_x), wrap_inc(M, i_y)][:]
        checks[i] += cl[wrap_inc(M, i_x), wrap_dec(M, i_y)][:]
        checks[i] += cl[wrap_dec(M, i_x), wrap_inc(M, i_y)][:]
        checks[i] += cl[wrap_dec(M, i_x), wrap_dec(M, i_y)][:]
    return checks

def get_inters(checks, r, r_cl, R_cl):
    R_cl_sq = R_cl ** 2
    inters = [[] for i in range(len(r))]
    for i in range(len(checks)):
        for i_cl in checks[i]:
            if utils.vector_mag_sq(r[i] - r_cl[i_cl]) < R_cl_sq[i_cl]:
                inters[i].append(i_cl)
    return inters