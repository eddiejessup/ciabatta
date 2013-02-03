import itertools
import numpy as np
import utils

filler = np.frompyfunc(lambda x: list(), 1, 1)

def get_cl_2d(r_cl, R_cl, L):
    R_cl_max = R_cl.max() if len(R_cl) > 0 else L
    M = int(L / R_cl_max)
    dx = L / M
    cl = np.empty([M, M], dtype=np.object)
    filler(cl, cl)
    inds = utils.r_to_i(r_cl, L, dx)
    for i in range(len(inds)):
        cl[tuple(inds[i])].append(i)
    return cl

def get_checks_2d(cl, r, L):
    M = cl.shape[0]
    dx = L / M
    inds = utils.r_to_i(r, L, dx)
    checks = np.empty([len(r)], dtype=np.object)

    xs, ys = inds.T
    xs_inc = np.where(xs < M - 1, xs + 1, 0)
    ys_inc = np.where(ys < M - 1, ys + 1, 0)
    xs_dec = np.where(xs > 0, xs - 1, M - 1)
    ys_dec = np.where(ys > 0, ys - 1, M - 1)
    for i in range(len(r)):
        x, y = xs[i], ys[i]
        x_inc, y_inc = xs_inc[i], ys_inc[i]
        x_dec, y_dec = xs_dec[i], ys_dec[i]

        checks[i] = itertools.chain(cl[x, y],
            cl[x_inc, y], cl[x_dec, y], cl[x, y_inc], cl[x, y_dec],
            cl[x_inc, y_inc], cl[x_inc, y_dec], cl[x_dec, y_inc], cl[x_dec, y_dec])
    return checks

def get_inters_2d(cl, r, L, r_cl, R_cl):
    R_cl_sq = R_cl ** 2
    checks = get_checks_2d(cl, r, L)
    inters = np.empty([len(r)], dtype=np.object)
    for i in range(len(r)):
        inters[i] = []
        for i_cl in checks[i]:
            if (r[i, 0] - r_cl[i_cl, 0]) ** 2 + (r[i, 1] - r_cl[i_cl, 1]) ** 2 < R_cl_sq[i_cl]:
                inters[i].append(i_cl)
    return inters

def get_inters(cl, r, L, r_cl, R_cl):
    if r.shape[-1] == 2: return get_inters_2d(cl, r, L, r_cl, R_cl)
    else: raise Exception('Dimension not supported for cell list')

def get_cl(r_cl, R_cl, L):
    if r_cl.shape[-1] == 2: return get_cl_2d(r_cl, R_cl, L)
    else: raise Exception('Dimension not supported for cell list')
