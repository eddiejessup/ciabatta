import timeit
import numpy as np
import matplotlib.pyplot as pp
import cell_list

n_runs = 20


def get_command(alg):
    if alg == 'p':
        return 'cell_list.interacts(r, L, R)'
    elif alg == 'pd':
        return 'cell_list.interacts_direct(r, L, R)'
    elif alg == 'f':
        return 'cell_list.interacts(r, L, R)'
    elif alg == 'fd':
        return 'cell_list.interacts_direct(r, L, R)'
    else:
        raise Exception('Invalid algorithm string')
    return command


def get_setup(R, L, N, dist):
    setup = ("" +
             "import numpy as np\n" +
             "import cell_list\n" +
             "R = %f\n" % R +
             "L = %f\n" % L +
             "L_half = L / 2.0\n")
    if dist == 'uniform':
        setup += 'r = np.random.uniform(-L_half, L_half, (%i, 2))\n' % N
    elif dist == 'point':
        setup += 'r = np.zeros((%i, 2))\n' % N
    else:
        raise Exception('Invalid distribution string')
    return setup


def timer(R, L, N, alg='p', dist='uniform'):
    return timeit.timeit(get_command(alg), setup=get_setup(R, L, N, dist), number=n_runs)

R_range = (0.001, 0.01)
N_range = (100, 10e3)
samples = 20


def time_surface(alg='cl', dist='uniform'):
    points, values = [], []
    Rs = np.logspace(np.log10(R_range[0]), np.log10(R_range[1]), num=samples)
    Ns = np.logspace(np.log10(N_range[0]), np.log10(N_range[1]), num=samples)
    ts = np.zeros([len(Rs), len(Ns)], dtype=np.float)
    for i in range(len(Rs)):
        for j in range(len(Ns)):
            ts[i, j] = timer(Rs[i], 1.0, Ns[j], alg=alg, dist=dist)
            print(Rs[i], Ns[j], ts[i, j])
    from mpl_toolkits.mplot3d import Axes3D
    fig = pp.figure()
    ax = fig.add_subplot(111, projection='3d')
    rs, ns = np.meshgrid(Rs, Ns)
    ax.plot_surface(rs, ns, ts.T)
    ax.set_xlabel('R')
    ax.set_ylabel('N')
    ax.set_zlabel('t')
    pp.show()


def test_consistency():
    R = 3
    L = 1200
    L_half = L / 2.0
    N = 5000
    r = np.random.uniform(-L_half, L_half, (N, 2))

#    p = cell_list.interacts(r, L, R)
#    pd = cell_list.interacts_direct(r, L, R)
    f_raw, f_lims = cell_list.interacts(r, L, R)
    fd_raw, fd_lims = cell_list.interacts_direct(r, L, R)

#    p = [sorted(entry) for entry in p]
#    pd = [sorted(entry) for entry in pd]
    f = [sorted(list(f_raw[i, :f_lims[i]] - 1)) for i in range(f_raw.shape[0])]
    fd = [sorted(list(fd_raw[i, :fd_lims[i]] - 1))
          for i in range(f_raw.shape[0])]

    if f == fd:
        print('All interacts functions equivalent!')
    else:
        for i in range(len(p)):
            if not (p[i] == pd[i] == f[i] == fd[i]):
                print(p[i], pd[i], f[i], fd[i])


def find_quickest(R, L, N, dist='uniform'):
    print('R: %f' % R)
    print('L: %f' % L)
    print('N: %i' % N)
    print('Distribution: %s' % dist)

    tp = timer(R, L, N, alg='p', dist=dist)
    tpd = timer(R, L, N, alg='pd', dist=dist)
    tf = timer(R, L, N, alg='f', dist=dist)
    tfd = timer(R, L, N, alg='fd', dist=dist)

    print('Python, Cell List: %f' % tp)
    print('Python, Direct: %f' % tpd)
    print('Fortran, Cell List: %f' % tf)
    print('Fortran, Direct: %f' % tfd)

if __name__ == '__main__':
#    test_consistency()
    find_quickest(3, 1200, 5000, 'uniform')
