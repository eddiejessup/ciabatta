'''
Pack spheres into periodic systems.
'''

import numpy as np
import geom
from periodic import pdist_periodic, cdist_periodic
import matplotlib.pyplot as plt


class MetroRCP(object):

    def __init__(self, r_0, L_0, R, dr_max, dL_max):
        self.r = r_0
        self.n, self.dim = self.r.shape
        self.L = L_0
        self.R = R
        self.dr_max = dr_max
        self.dL_max = dL_max

        self.sep_sq = pdist_periodic(r_0, self.L)

    def U(self):
        if np.any(self.sep_sq < (2.0 * self.R) ** 2):
            return np.inf
        return 1.0 / self.pf()

    def displace_r(self):
        self.i = np.random.randint(self.n)
        self.r_old = self.r[self.i].copy()

        dr = np.random.uniform(-self.dr_max * self.L,
                               self.dr_max * self.L, self.dim)
        self.r[self.i] += dr
        self.r[self.r > self.L / 2.0] -= self.L
        self.r[self.r < -self.L / 2.0] += self.L

        self.sep_sq_old = self.sep_sq[self.i].copy()

        sep_sq = cdist_periodic(self.r[np.newaxis, self.i], self.r, self.L)

        self.sep_sq[self.i, :] = sep_sq
        self.sep_sq[:, self.i] = sep_sq
        self.sep_sq[self.i, self.i] = np.inf

    def revert_r(self):
        self.r[self.i] = self.r_old.copy()

        self.sep_sq[self.i, :] = self.sep_sq_old.copy()
        self.sep_sq[:, self.i] = self.sep_sq_old.copy()

    def displace_L(self):
        self.dL = 1.0 + np.random.uniform(-self.dL_max, self.dL_max)

        self.L *= self.dL
        self.r *= self.dL
        self.sep_sq *= self.dL ** 2

    def revert_L(self):
        self.L /= self.dL
        self.r /= self.dL
        self.sep_sq /= self.dL ** 2

    def iterate(self, beta):
        U_0 = self.U()

        i = np.random.randint(self.n + 1)
        if i < len(self.r):
            self.displace_r()
            revert = self.revert_r
        else:
            self.displace_L()
            revert = self.revert_L

        U_new = self.U()
        if np.exp(-beta * (U_new - U_0)) < np.random.uniform():
            revert()

    def V(self):
        return self.L ** self.dim

    def V_full(self):
        return self.n * geom.sphere_volume(self.R, self.dim)

    def pf(self):
        return self.V_full() / self.V()


def n_to_pf(L, d, n, R):
    '''
    Packing fraction of n spheres
    in d dimensions
    of radius R
    in a system of length L.
    '''
    return (n * geom.sphere_volume(R=R, n=d)) / L ** d


def pf_to_n(L, d, pf, R):
    '''
    Number of spheres of radius R
    to achieve as close to packing fraction pf as possible
    in d dimensions
    in a system of length L.
    Also return the actual achieved packing fraction.
    '''
    n = int(round(pf * L ** d / geom.sphere_volume(R, d)))
    pf_actual = n_to_pf(L, d, n, R)
    return n, pf_actual


def calc_L_0(n, d, pf, R):
    '''
    Calculate system size required such that
    for n spheres of radius R in d dimensions,
    the packing fraction is pf.
    Useful to initialise Metropolis algorithm to a reasonable state.
    '''
    return ((n * geom.sphere_volume(R=R, n=d)) / pf) ** (1.0 / d)


def pack_simple(L, d,
                n, R, seed):
    '''
    Pack n spheres into a system
    of length L, dimension d,
    using random seed,
    using naïve uniform distribution of spheres,
    and the tabula rasa rule.
    (Start over from scratch if overlapping occurs.)
    '''
    np.random.seed(seed)
    while True:
        r = np.random.uniform(-L / 2.0, L / 2.0, size=(n, d))
        if not np.any(pdist_periodic(r, L) < (2.0 * R) ** 2):
            return r

every = 5000


def unwrap_one_layer(r, L, n):
    if n == 0:
        return list(r)
    rcu = []
    for x, y in r:
        for ix in range(-n, n + 1):
            for iy in range(-n, n + 1):
                if abs(ix) == n or abs(iy) == n:
                    rcu.append(np.array([x + ix * L, y + iy * L]))
    return rcu


def unwrap_to_layer(r, L, n=1):
    rcu = []
    for i_n in range(n + 1):
        rcu.extend(unwrap_one_layer(r, L, i_n))
    return rcu


def draw_medium(r, R, L, n=1, ax=None):
    if ax is None:
        ax = plt.gca()
    for ru in unwrap_to_layer(r, L, n):
        c = plt.Circle(ru, radius=R, alpha=0.2)
        print(ru, R)
        ax.add_artist(c)
    # ax.set_aspect('equal')


def pack(dim, R, beta_max=1e4, dL_max=0.02, dr_max=0.02,
         seed=None, pf=None, n=None):
    '''
    Create a configuration of packed spheres
    with radius R (expressed as a fraction of the system size).
    in a periodic system of dimension dim.

    Sphere packing can be specified by volume fraction vf,
    or number of spheres n.

    Configuration is created through the Metropolis-Hastings
    algorithm for an NPT system.
    '''

    if pf is not None:
        if pf == 0.0:
            return np.array([]), R
        # If packing fraction is specified, find required number of spheres
        # and the actual packing fraction this will produce
        n, pf_actual = pf_to_n(1.0, dim, pf, R)
    elif n is not None:
        if n == 0:
            return np.array([]), R
        # If n is specified, find packing fraction
        pf_actual = n_to_pf(1.0, dim, n, R)

    # Calculate an initial packing fraction and system size
    # Start at at most 0.5%; lower if the desired packing fraction is very low
    pf_initial = min(0.005, pf_actual / 2.0)
    # Find system size that will create this packing fraction
    L_0 = calc_L_0(n, dim, pf_initial, R)

    # Pack naïvely into this system
    r_0 = pack_simple(L_0, dim, n, R, seed)
    print('Initial packing done')

    mg = MetroRCP(r_0, L_0, R, dr_max, dL_max)

    t = 0
    while mg.pf() < pf_actual:
        t += 1
        beta = beta_max * mg.pf()
        mg.iterate(beta)

        if not t % every:
            print('Packing: {:.1f}%'.format(100.0 * mg.pf()))

    # print('Final packing: {:.1f}%'.format(100.0 * mg.pf()))

    return mg.r / mg.L, mg.R / mg.L
