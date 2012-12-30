import timeit
import numpy as np
import matplotlib.pyplot as pp
import cell_list

n_runs = 20

def test_consistency():
    R = 3
    L = 1200
    L_half = L / 2.0
    N = 5000
    r = np.random.uniform(-L_half, L_half, (N, 2))

    p = cell_list.interacts(r, L, R)
    pd = cell_list.interacts_direct(r, L, R)
    f_raw, f_lims = cell_list.interacts_fort(r, L, R)
    fd_raw, fd_lims = cell_list.interacts_fort_direct(r, L, R)

    p = [sorted(entry) for entry in p]
    pd = [sorted(entry) for entry in pd]
    f = [sorted(list(f_raw[i, :f_lims[i]] - 1)) for i in range(f_raw.shape[0])]
    fd = [sorted(list(fd_raw[i, :fd_lims[i]] - 1)) for i in range(f_raw.shape[0])]

    if p == pd == f == fd:
        print('All interacts functions equivalent!')
        return

    for i in range(len(d)):
        if not (p[i] == pd[i] == f[i] == fd[i]): print(p[i], pd[i], f[i], fd[i])

def find_quickest(R, L, N, dist='uniform'):
    print('R: %f' % R)
    print('L: %f' % L)
    print('N: %i' % N)
    print('Distribution: %s' % dist)
    setup = '''
import numpy as np, cell_list
R = %f
L = %f
L_half = L / 2.0
''' % (R, L)

    if dist == 'uniform':
        setup += 'r = np.random.uniform(-L_half, L_half, (%i, 2))' % N
    elif dist == 'point':
        setup += 'r = np.zeros((%i, 2))' % N

    L_half = L / 2.0
    r = np.random.uniform(-L_half, L_half, (N, 2))

    tp = timeit.timeit('cell_list.interacts(r, L, R)', setup=setup, number=n_runs)
    tpd = timeit.timeit('cell_list.interacts_direct(r, L, R)', setup=setup, number=n_runs)
    tf = timeit.timeit('cell_list.interacts_fort(r, L, R)', setup=setup, number=n_runs)
    tfd = timeit.timeit('cell_list.interacts_fort_direct(r, L, R)', setup=setup, number=n_runs)
    print('Python, Cell List: %f' % tp)
    print('Python, Direct: %f' % tpd)
    print('Fortran, Cell List: %f' % tf)
    print('Fortran, Direct: %f' % tfd)

if __name__ == '__main__':
#    time_check()
    test_consistency()
    find_quickest(3.0, 1200, 1000, 'uniform')
