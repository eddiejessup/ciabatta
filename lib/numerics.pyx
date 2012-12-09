# cython: profile=True

import numpy as np
import fields
import utils

cimport numpy as np
from utils_cy cimport square, wrap_inc, wrap_dec, wrap_real

def interacts_cell_list(np.ndarray[np.float_t, ndim=2] r, double L, double R_cut):
    cdef unsigned int i, i_2, i_dim
    cdef double R_cut_sq = square(R_cut), L_half = L / 2.0, R_sep_sq
    cdef list interacts = [[] for i in range(r.shape[0])]

    # Cell list stuff
    cdef unsigned int M = int(L / R_cut)
    cdef list checks = [[] for i in range(r.shape[0])]
    cdef list cell_list = utils.empty_lists(M, r.shape[1])
    cdef np.ndarray[np.int_t, ndim=2] inds = np.asarray((r + L_half) /  (L / float(M)), dtype=np.int)

    if r.shape[1] == 1:
        for i in range(r.shape[0]):
            cell_list[inds[i, 0]].append(i)

        for i in range(r.shape[0]):
            i_x, = inds[i]
            checks[i] += cell_list[i_x]
            checks[i] += cell_list[wrap_inc(M, i_x)]
            checks[i] += cell_list[wrap_dec(M, i_x)]

    elif r.shape[1] == 2:
        for i in range(r.shape[0]):
            cell_list[inds[i, 0]][inds[i, 1]].append(i)

        for i in range(r.shape[0]):
            i_x, i_y = inds[i]
            checks[i] += cell_list[i_x][i_y]
            checks[i] += cell_list[wrap_inc(M, i_x)][i_y]
            checks[i] += cell_list[wrap_dec(M, i_x)][i_y]
            checks[i] += cell_list[i_x][wrap_inc(M, i_y)]
            checks[i] += cell_list[i_x][wrap_dec(M, i_y)]
            checks[i] += cell_list[wrap_inc(M, i_x)][wrap_inc(M, i_y)]
            checks[i] += cell_list[wrap_inc(M, i_x)][wrap_dec(M, i_y)]
            checks[i] += cell_list[wrap_dec(M, i_x)][wrap_inc(M, i_y)]
            checks[i] += cell_list[wrap_dec(M, i_x)][wrap_dec(M, i_y)]

    elif r.shape[1] == 3:
        for i in range(r.shape[0]):
            cell_list[inds[i, 0]][inds[i, 1]][inds[i, 2]].append(i)
        raise NotImplementedError

    else:
        raise Exception('Cell list not implemented in this dimension')

    for i in range(r.shape[0]):
        checks[i].remove(i)       
        for i_2 in checks[i]:
            R_sep_sq = 0.0
            for i_dim in range(r.shape[1]):
                R_sep_sq += square(wrap_real(L, L_half, r[i, i_dim] - r[i_2, i_dim]))
            if R_sep_sq < R_cut_sq:
                interacts[i].append(i_2)
    return interacts

def interacts_direct(np.ndarray[np.float_t, ndim=2] r, double L, double R_cut):
    cdef unsigned int i_1, i_2, i_dim
    cdef double R_cut_sq = square(R_cut), L_half = L / 2.0, R_sep_sq
    cdef list interacts = [[] for i_1 in range(r.shape[0])]

    for i_1 in range(r.shape[0]):
        for i_2 in range(i_1 + 1, r.shape[0]):
            R_sep_sq = 0.0
            for i_dim in range(r.shape[1]):
                R_sep_sq += square(wrap_real(L, L_half, r[i_1, i_dim] - r[i_2, i_dim]))
            if R_sep_sq < R_cut_sq:
                interacts[i_1].append(i_2)
                interacts[i_2].append(i_1)
    return interacts

def r_sep(np.ndarray[np.float_t, ndim=2] r, double L):
    cdef unsigned int i_1, i_2, i_dim
    cdef double L_half = L / 2.0

    cdef np.ndarray[np.float_t, ndim = 3] r_sep = \
        np.zeros(2 * (r.shape[0],) + (r.shape[1],), dtype=np.float)
    cdef np.ndarray[np.float_t, ndim = 2] R_sep_sq = \
        np.zeros(2 * (r.shape[0],), dtype=np.float)

    for i_1 in range(r.shape[0]):
        for i_2 in range(i_1 + 1, r.shape[0]):
            for i_dim in range(r.shape[1]):
                r_sep[i_1, i_2, i_dim] = wrap_real(L, L_half, r[i_2, i_dim] - r[i_1, i_dim])
                r_sep[i_2, i_1, i_dim] = -r_sep[i_1, i_2, i_dim]
                R_sep_sq[i_1, i_2] += square(r_sep[i_1, i_2, i_dim])
            R_sep_sq[i_2, i_1] = R_sep_sq[i_1, i_2]
    return r_sep, R_sep_sq
