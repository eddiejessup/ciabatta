'''
For a set of points in a periodic system,
find those within a cut-off distance of one another.
'''

from ciabatta.cell_list import _intro


def get_inters(r, L, R_cut):
    '''
    Returns points within a given cut-off of each other,
    in a periodic system.

    Uses a cell-list.

    Parameters
    ----------
    r: array, shape (n, d) where d is one of (2, 3).
        A set of n point coordinates.
        Coordinates are assumed to lie in [-L / 2, L / 2].
    L: float.
        Bounds of the system.
    R_cut: float.
        The maximum distance within which to consider points to lie
        near each other.

    Returns
    -------
    inters: array, shape (n, n)
        Indices of the nearby points.
        For each particle indexed by the first axis,
        the second axis lists each index of a nearby point.
    intersi: array, shape (n,)
        Total number of nearby points.
        This array should be used to index `inters`, as for point `i`,
        elements in `inters[i]` beyond `intersi[i]` have no well-defined value.
    '''
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
    '''
    Returns points within a given cut-off of each other,
    in a periodic system.

    Uses a direct algorithm, which may be very slow for large numbers of
    points.

    Parameters
    ----------
    r: array, shape (n, d) where d is one of (2, 3).
        A set of n point coordinates.
        Coordinates are assumed to lie in [-L / 2, L / 2].
    L: float.
        Bounds of the system.
    R_cut: float.
        The maximum distance within which to consider points to lie
        near each other.

    Returns
    -------
    inters: array, shape (n, n)
        Indices of the nearby points.
        For each particle indexed by the first axis,
        the second axis lists each index of a nearby point.
    intersi: array, shape (n,)
        Total number of nearby points.
        This array should be used to index `inters`, as for point `i`,
        elements in `inters[i]` beyond `intersi[i]` have no well-defined value.
    '''
    _intro.cell_list_direct.make_inters(r.T, L, R_cut)
    return parse_inters()


def parse_inters():
    return (_intro.cell_list_shared.inters.T - 1,
            _intro.cell_list_shared.intersi.T)
