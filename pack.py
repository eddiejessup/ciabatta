'''
Pack spheres into periodic systems.
'''

import numpy as np
from ciabatta import geom
from ciabatta.distance import pdist_sq_periodic, cdist_sq_periodic
import matplotlib.pyplot as plt

every = 5000


class MetroRCP(object):

    def __init__(self, r_0, L_0, R, dr_max, dL_max):
        self.r = r_0
        self.n, self.dim = self.r.shape
        self.L = L_0
        self.R = R
        self.dr_max = dr_max
        self.dL_max = dL_max

        self.sep_sq = pdist_sq_periodic(r_0, self.L)

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

        sep_sq = cdist_sq_periodic(self.r[np.newaxis, self.i], self.r, self.L)

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


def unwrap_one_layer(r, L, n):
    '''
    For a set of points in a 2 dimensional periodic system, extend the set of
    points to tile the points at a given period.

    Parameters
    ----------
    r: float array, shape (:, 2).
        Set of points.
    L: float.
        System size.
    n: integer.
        Period to unwrap.

    Returns
    -------
    rcu: float array, shape (:, 2).
        The set of points. tiled at the periods at a distance `n` from the
        origin.
    '''
    try:
        L[0]
    except (TypeError, IndexError):
        L = np.ones([r.shape[1]]) * L
    if n == 0:
        return list(r)
    rcu = []
    for x, y in r:
        for ix in range(-n, n + 1):
            for iy in range(-n, n + 1):
                if abs(ix) == n or abs(iy) == n:
                    rcu.append(np.array([x + ix * L[0], y + iy * L[1]]))
    return rcu


def unwrap_to_layer(r, L, n=1):
    '''
    For a set of points in a 2 dimensional periodic system, extend the set of
    points to tile the points up to to a given period.

    Parameters
    ----------
    r: float array, shape (:, 2).
        Set of points.
    L: float.
        System size.
    n: integer.
        Period to unwrap up to.

    Returns
    -------
    rcu: float array, shape (:, 2).
        The set of points. tiled up to the periods at a distance `n` from the
        origin.
    '''
    rcu = []
    for i_n in range(n + 1):
        rcu.extend(unwrap_one_layer(r, L, i_n))
    return rcu


def draw_medium(r, R, L, n=1, ax=None):
    '''
    Draw circles representing circles in a two-dimensional periodic system.
    Circles may be tiled up to a number of periods.

    Parameters
    ----------
    r: float array, shape (:, 2).
        Set of points.
    R: float
        Circle radius.
    L: float.
        System size.
    n: integer.
        Period to unwrap up to.
    ax: matplotlib axes instance or None
        Axes to draw circles onto. If `None`, use default axes.

    Returns
    -------
    None
    '''
    if ax is None:
        ax = plt.gca()
    for ru in unwrap_to_layer(r, L, n):
        c = plt.Circle(ru, radius=R, alpha=0.2)
        ax.add_artist(c)


def n_to_pf(L, d, n, R):
    '''
    Returns the packing fraction for a number of non-intersecting spheres.

    Parameters
    ----------
    L: float
        System length.
    d: integer
        System dimension.
    n: integer
        Number of spheres.
    R: float
        Sphere radius.

    Returns
    -------
    pf: float
        Fraction of space occupied by the spheres.
    '''
    return (n * geom.sphere_volume(R=R, n=d)) / L ** d


def pf_to_n(L, d, pf, R):
    '''
    Returns the number of non-intersecting spheres required to achieve
    as close to a given packing fraction as possible, along with the actual
    achieved packing fraction. for a number of non-intersecting spheres.

    Parameters
    ----------
    L: float
        System length.
    d: integer
        System dimension.
    pf: float
        Fraction of space to be occupied by the spheres.
    R: float
        Sphere radius.

    Returns
    -------
    n: integer
        Number of spheres required to achieve a packing fraction `pf_actual`
    pf_actual:
        Fraction of space occupied by `n` spheres.
        This is the closest possible fraction achievable to `pf`.
    '''
    n = int(round(pf * L ** d / geom.sphere_volume(R, d)))
    pf_actual = n_to_pf(L, d, n, R)
    return n, pf_actual


def calc_L_0(n, d, pf, R):
    '''
    Returns the system size required to achieve a given packing fraction,
    for a number of non-intersecting spheres.

    Useful to initialise the Metropolis algorithm to a reasonable state.

    Parameters
    ----------
    n: integer
        Number of spheres.
    d: integer
        System dimension.
    pf: float
        Fraction of space to be occupied by the spheres.
    R: float
        Sphere radius.

    Returns
    -------
    L_0: float
        System size.
    '''
    return ((n * geom.sphere_volume(R=R, n=d)) / pf) ** (1.0 / d)


def pack_simple(d, R, L, seed=None, pf=None, n=None):
    '''
    Pack a number of non-intersecting spheres into a periodic system.

    Can specify packing by number of spheres or packing fraction.

    This implementation uses a naive uniform distribution of spheres,
    and the Tabula Rasa rule (start from scratch if an intersection occurs).

    This is likely to be very slow for high packing fractions

    Parameters
    ----------
    d: integer
        System dimension.
    R: float
        Sphere radius.
    L: float
        System size.
    seed: integer or None.
        Seed for the random number generator.
        None will use a different seed for each call.
    pf: float or None
        Packing fraction
    n: integer or None
        Number of spheres.

    Returns
    -------
    r: float array, shape (n, d)
        Coordinates of the centres of the spheres for a valid configuration.
    R_actual: float
        Actual sphere radius used in the packing.
        In this implementation this will always be equal to `R`;
        it is returned only to provide a uniform interface with the
        Metropolis-Hastings implementation.
    '''
    np.random.seed(seed)

    if pf is not None:
        if pf == 0.0:
            return np.array([]), R
        # If packing fraction is specified, find required number of spheres
        # and the actual packing fraction this will produce
        n, pf_actual = pf_to_n(L, d, pf, R)
    elif n is not None:
        if n == 0:
            return np.array([]), R

    while True:
        r = np.random.uniform(-L / 2.0, L / 2.0, size=(n, d))
        if not np.any(pdist_sq_periodic(r, L) < (2.0 * R) ** 2):
            return r, R


def pack(d, R, L, seed=None, pf=None, n=None,
         beta_max=1e4, dL_max=0.02, dr_max=0.02):
    '''
    Pack a number of non-intersecting spheres into a periodic system.

    Can specify packing by number of spheres or packing fraction.

    This implementation uses the Metropolis-Hastings algorithm for an
    NPT system.

    Parameters
    ----------
    d: integer
        System dimension.
    R: float
        Sphere radius.
    L: float
        System size.
    seed: integer or None.
        Seed for the random number generator.
        None will use a different seed for each call.
    pf: float or None
        Packing fraction
    n: integer or None
        Number of spheres.

    Metropolis-Hastings parameters
    ------------------------------
    Playing with these parameters may improve packing speed.

    beta_max: float, greater than zero.
        Inverse temperature which controls how little noiseis in the system.
    dL_max: float, 0 < dL_max < 1
        Maximum fraction by which to perturb the system size.
    dr_max: float, 0 < dr_max < 1
        Maximum system fraction by which to perturb sphere positions.

    Returns
    -------
    r: float array, shape (n, d)
        Coordinates of the centres of the spheres for a valid configuration.
    R_actual: float
        Actual sphere radius used in the packing.
    '''
    np.random.seed(seed)
    if pf is not None:
        if pf == 0.0:
            return np.array([]), R
        # If packing fraction is specified, find required number of spheres
        # and the actual packing fraction this will produce
        n, pf_actual = pf_to_n(L, d, pf, R)
    elif n is not None:
        if n == 0:
            return np.array([]), R
        # If n is specified, find packing fraction
        pf_actual = n_to_pf(L, d, n, R)

    # Calculate an initial packing fraction and system size
    # Start at at most 0.5%; lower if the desired packing fraction is very low
    pf_initial = min(0.005, pf_actual / 2.0)
    # Find system size that will create this packing fraction
    L_0 = calc_L_0(n, d, pf_initial, R)

    # Pack naively into this system
    r_0, R = pack_simple(d, R, L_0, seed, n=n)
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

    return mg.r, mg.R
