# cython: profile=False
import numpy as np
import _cell_list

cimport numpy as np
from utils_cy cimport square, wrap_inc, wrap_dec, wrap_real

cdef double R_sep_sq(double r_1_x, double r_1_y, double r_2_x, double r_2_y, double L, double L_half):
    return square(wrap_real(L, L_half, r_1_x - r_2_x)) + square(wrap_real(L, L_half, r_1_y - r_2_y))

def interacts_fort(r, L, R_cut):
    if r.shape[1] == 2:
        _cell_list.cell_list_2d.make_inters(r.T, L, R_cut)
        inters, intersi =  _cell_list.cell_list_2d.inters.T, _cell_list.cell_list_2d.intersi.T
    else:
        raise Exception('Cell list not implemented in this dimension')
    return inters, intersi

def interacts_fort_direct(r, L, R_cut):
    if r.shape[1] == 2:
        _cell_list.cell_list_2d.make_inters_direct(r.T, L, R_cut)
        inters, intersi =  _cell_list.cell_list_2d.inters.T, _cell_list.cell_list_2d.intersi.T
    else:
        raise Exception('Inters not implemented in this dimension')
    return inters, intersi

def interacts(np.ndarray[np.float_t, ndim=2] r, double L, double R_cut):
    cdef unsigned int M = int(L / R_cut), i, i_2, x, y, x_inc, x_dec, y_inc, y_dec
    cdef double R_cut_sq = square(R_cut), L_half = L / 2.0
    cdef list cell_list = [[[] for i_2 in range(M)] for i in range(M)]
    cdef list interacts = [[] for i in range(r.shape[0])]

    cdef np.ndarray[np.int_t, ndim=2] inds = np.asarray((r + L_half) /  (L / float(M)), dtype=np.int)

    for i in range(r.shape[0]):
        cell_list[inds[i, 0]][inds[i, 1]].append(i)

    for x in range(M):
        for y in range(M):
            if not cell_list[x][y]: continue
            x_inc = wrap_inc(M, x)
            x_dec = wrap_dec(M, x)
            y_inc = wrap_inc(M, y)
            y_dec = wrap_dec(M, y)
            for i in cell_list[x][y]:
                for i_2 in cell_list[x][y]:
                    if i_2 == i: continue
                    if R_sep_sq(r[i, 0], r[i, 1], r[i_2, 0], r[i_2, 1], L, L_half) < R_cut_sq:
                        interacts[i].append(i_2)
                for i_2 in cell_list[x][y_inc]:
                    if R_sep_sq(r[i, 0], r[i, 1], r[i_2, 0], r[i_2, 1], L, L_half) < R_cut_sq:                    
                        interacts[i].append(i_2)
                for i_2 in cell_list[x][y_dec]:
                    if R_sep_sq(r[i, 0], r[i, 1], r[i_2, 0], r[i_2, 1], L, L_half) < R_cut_sq:
                        interacts[i].append(i_2)
                for i_2 in cell_list[x_inc][y]:
                    if R_sep_sq(r[i, 0], r[i, 1], r[i_2, 0], r[i_2, 1], L, L_half) < R_cut_sq:
                        interacts[i].append(i_2)
                for i_2 in cell_list[x_dec][y]:
                    if R_sep_sq(r[i, 0], r[i, 1], r[i_2, 0], r[i_2, 1], L, L_half) < R_cut_sq:
                        interacts[i].append(i_2)
                for i_2 in cell_list[x_inc][y_inc]:
                    if R_sep_sq(r[i, 0], r[i, 1], r[i_2, 0], r[i_2, 1], L, L_half) < R_cut_sq:
                        interacts[i].append(i_2)
                for i_2 in cell_list[x_inc][y_dec]:
                    if R_sep_sq(r[i, 0], r[i, 1], r[i_2, 0], r[i_2, 1], L, L_half) < R_cut_sq:
                        interacts[i].append(i_2)
                for i_2 in cell_list[x_dec][y_inc]:
                    if R_sep_sq(r[i, 0], r[i, 1], r[i_2, 0], r[i_2, 1], L, L_half) < R_cut_sq:
                        interacts[i].append(i_2)
                for i_2 in cell_list[x_dec][y_dec]:
                    if R_sep_sq(r[i, 0], r[i, 1], r[i_2, 0], r[i_2, 1], L, L_half) < R_cut_sq:
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
