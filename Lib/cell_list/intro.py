'''
This is for an introspective point set, i.e. for a set of dynamic points, seeing
which are within a cut-off distance of one another.
'''

from cell_list import _intro

def get_inters(r, L, R_cut):
    if r.shape[1] == 2:
        _intro.cell_list_2d.make_inters(r.T, L, R_cut)
        inters, intersi = _intro.cell_list_2d.inters.T, _intro.cell_list_2d.intersi.T
    else:
        raise Exception('Inters cell list not implemented in this dimension')
    return inters, intersi

def get_inters_direct(r, L, R_cut):
    if r.shape[1] == 2:
        _intro.cell_list_2d.make_inters_direct(r.T, L, R_cut)
        inters, intersi =  _intro.cell_list_2d.inters.T, _intro.cell_list_2d.intersi.T
    else:
        raise Exception('Inters direct not implemented in this dimension')
    return inters, intersi
