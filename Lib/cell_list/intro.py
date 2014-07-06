'''
This is for an introspective point set, i.e. for a set of dynamic points, seeing
which are within a cut-off distance of one another.
'''

from cell_list import _intro

def get_inters(r, L, R_cut):
    if r.shape[1] == 2:
        _intro.cell_list_2d.make_inters(r.T, L, R_cut)
    elif r.shape[1] == 3:
        _intro.cell_list_3d.make_inters(r.T, L, R_cut)
    else:
        print('Warning: cell list not implemented in this dimension, falling'
              'back to direct computation')
        return get_inters_direct(r, L, R_cut)
    return parse_inters()

def get_inters_direct(r, L, R_cut):
    _intro.cell_list_direct.make_inters(r.T, L, R_cut)
    return parse_inters()

def parse_inters():
    return _intro.cell_list_shared.inters.T - 1, _intro.cell_list_shared.intersi.T