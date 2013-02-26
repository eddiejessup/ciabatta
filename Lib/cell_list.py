'''
Note, this is for an introspective cell list, i.e. for a set of points, seeing
which are within a cut-off distance of one another. See "cl" for the case where
there are two distinct sets of points for comparison (e.g. particles and
obstacles).
'''

import _cell_list

def interacts(r, L, R_cut):
    if r.shape[1] == 2:
        _cell_list.cell_list_2d.make_inters(r.T, L, R_cut)
        inters, intersi = _cell_list.cell_list_2d.inters.T, _cell_list.cell_list_2d.intersi.T
    else:
        raise Exception('Inters cell list not implemented in this dimension')
    return inters, intersi

def interacts_direct(r, L, R_cut):
    if r.shape[1] == 2:
        _cell_list.cell_list_2d.make_inters_direct(r.T, L, R_cut)
        inters, intersi =  _cell_list.cell_list_2d.inters.T, _cell_list.cell_list_2d.intersi.T
    else:
        raise Exception('Inters direct not implemented in this dimension')
    return inters, intersi
