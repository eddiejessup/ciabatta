import timeit
import numpy as np
import matplotlib.pyplot as pp
import cell_list

n_runs = 20
N_growth = 1.2

def time_check():
    R = 1
    L = 1200
    L_half = L / 2.0
    N = 500
    Ns, tds, tcs, tns = [], [], [], []
    while True:
        setup = '''
import numpy as np, numerics
R = %f
L = %f
r = np.random.uniform(-%f, %f, (%i, 2))
''' % (R, L, L_half, L_half, N)
        td = timeit.timeit('cell_list.interacts_direct(r, L, R)', setup=setup, number=n_runs)
        tds.append(td)

        tc = timeit.timeit('cell_list.interacts_cl_checks(r, L, R)', setup=setup, number=n_runs)
        tcs.append(tc)

        tn = timeit.timeit('cell_list.interacts_cl(r, L, R)', setup=setup, number=n_runs)
        tns.append(tn)

        print(N)
        Ns.append(N)
        N *= N_growth
        if max(tc, tn) > 5: break

    pp.plot(Ns, tds)
    pp.plot(Ns, tcs)
    pp.plot(Ns, tns)
    pp.show()

def test_consistency():
    R = 3
    L = 1200
    L_half = L / 2.0
    N = 1000
    r = np.random.uniform(-L_half, L_half, (N, 2))

    d = cell_list.interacts_direct(r, L, R)
    c = cell_list.interacts_cl_checks(r, L, R)
    n = cell_list.interacts_cl(r, L, R)

    d = [sorted(entry) for entry in d]
    c = [sorted(entry) for entry in c]
    n = [sorted(entry) for entry in n]

    if d == c == n: 
        print('All interacts functions equivalent!')
        return

    for i in range(len(d)):
        if d[i] != c[i] or d[i] != c[i] or c[i] != n[i]: print(d[i], c[i], n[i])

def find_quickest(R, L, N, dist='uniform'):
    print('R: %f' % R)
    print('L: %f' % L)
    print('N: %i' % N)
    print('Distribution: %s' % dist)
    setup = '''
import numpy as np, numerics
R = %f
L = %f
L_half = L / 2.0
''' % (R, L)

    if dist == 'uniform':
        setup += 'r = np.random.uniform(-L_half, L_half, (%i, 2))' % N
    elif dist == 'point':
        setup += 'r = np.zeros((%i, 2))' % N
        
    td = timeit.timeit('cell_list.interacts_direct(r, L, R)', setup=setup, number=n_runs)
    tc = timeit.timeit('cell_list.interacts_cl_checks(r, L, R)', setup=setup, number=n_runs)
    tn = timeit.timeit('cell_list.interacts_cl(r, L, R)', setup=setup, number=n_runs)
    print('Direct: %f' % td)
    print('Cell List, checks: %f' % tc)
    print('Cell List, no checks: %f' % tn)
    if min(td, tc, tn) == td: print('Direct')
    elif min(td, tc, tn) == tc: print('Cell list using checks')
    elif min(td, tc, tn) == tn: print('Cell list without checks')

if __name__ == '__main__':
#    time_check()
#    test_consistency()
    find_quickest(0.5, 1200, 5000, 'point')
