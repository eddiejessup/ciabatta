"""
Pack spheres into periodic systems.
"""

from __future__ import print_function, division
import numpy as np
from ciabatta import geom
from ciabatta.distance import pdist_sq_periodic, cdist_sq_periodic
import matplotlib.pyplot as plt

every = 5000


class MetroRCP(object):

    def __init__(self, r_0, L_0, R, dr_max, dL_max, rng=None):
        self.r = r_0
        self.n, self.dim = self.r.shape
        self.L = L_0
        self.R = R
        self.dr_max = dr_max
        self.dL_max = dL_max
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng

        self.sep_sq = pdist_sq_periodic(r_0, self.L)

    def U(self):
        if np.any(self.sep_sq < (2.0 * self.R) ** 2):
            return np.inf
        return 1.0 / self.pf()

    def displace_r(self):
        self.i = self.rng.randint(self.n)
        self.r_old = self.r[self.i].copy()

        dr = np.zeros([self.dim])
        for i_dim in range(self.dim):
            dr[i_dim] = self.rng.uniform(-self.dr_max * self.L[i_dim],
                                         self.dr_max * self.L[i_dim])

        self.r[self.i] += dr
        for i_dim in range(self.dim):
            r = self.r[:, i_dim]
            r[r > self.L[i_dim] / 2.0] -= self.L[i_dim]
            r[r < -self.L[i_dim] / 2.0] += self.L[i_dim]

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
        self.dL = 1.0 + self.rng.uniform(-self.dL_max, self.dL_max)

        self.L *= self.dL
        self.r *= self.dL
        self.sep_sq *= self.dL ** 2

    def revert_L(self):
        self.L /= self.dL
        self.r /= self.dL
        self.sep_sq /= self.dL ** 2

    def iterate(self, beta):
        U_0 = self.U()

        i = self.rng.randint(self.n + 1)
        if i < len(self.r):
            self.displace_r()
            revert = self.revert_r
        else:
            self.displace_L()
            revert = self.revert_L

        U_new = self.U()
        if np.exp(-beta * (U_new - U_0)) < self.rng.uniform():
            revert()

    def V(self):
        return np.product(self.L)

    def V_full(self):
        return self.n * geom.sphere_volume(self.R, self.dim)

    def pf(self):
        return self.V_full() / self.V()


def unwrap_one_layer(r, L, n):
    """For a set of points in a 2 dimensional periodic system, extend the set of
    points to tile the points at a given period.

    Parameters
    ----------
    r: float array, shape (:, 2).
        Set of points.
    L: float array, shape (2,)
        System lengths.
    n: integer.
        Period to unwrap.

    Returns
    -------
    rcu: float array, shape (:, 2).
        The set of points. tiled at the periods at a distance `n` from the
        origin.
    """
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
    """For a set of points in a 2 dimensional periodic system, extend the set of
    points to tile the points up to to a given period.

    Parameters
    ----------
    r: float array, shape (:, 2).
        Set of points.
    L: float array, shape (2,)
        System lengths.
    n: integer.
        Period to unwrap up to.

    Returns
    -------
    rcu: float array, shape (:, 2).
        The set of points. tiled up to the periods at a distance `n` from the
        origin.
    """
    rcu = []
    for i_n in range(n + 1):
        rcu.extend(unwrap_one_layer(r, L, i_n))
    return rcu


def draw_medium(r, R, L, n=1, ax=None):
    """Draw circles representing circles in a two-dimensional periodic system.
    Circles may be tiled up to a number of periods.

    Parameters
    ----------
    r: float array, shape (:, 2).
        Set of points.
    R: float
        Circle radius.
    L: float array, shape (2,)
        System lengths.
    n: integer.
        Period to unwrap up to.
    ax: matplotlib axes instance or None
        Axes to draw circles onto. If `None`, use default axes.

    Returns
    -------
    None
    """
    if ax is None:
        ax = plt.gca()
    for ru in unwrap_to_layer(r, L, n):
        c = plt.Circle(ru, radius=R, alpha=0.2)
        ax.add_artist(c)


def n_to_pf(L, n, R):
    """Returns the packing fraction for a number of non-intersecting spheres.

    Parameters
    ----------
    L: float array, shape (d,)
        System lengths.
    n: integer
        Number of spheres.
    R: float
        Sphere radius.

    Returns
    -------
    pf: float
        Fraction of space occupied by the spheres.
    """
    dim = L.shape[0]
    return (n * geom.sphere_volume(R=R, n=dim)) / np.product(L)


def pf_to_n(L, pf, R):
    """Returns the number of non-intersecting spheres required to achieve
    as close to a given packing fraction as possible, along with the actual
    achieved packing fraction. for a number of non-intersecting spheres.

    Parameters
    ----------
    L: float array, shape (d,)
        System lengths.
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
    """
    dim = L.shape[0]
    n = int(round(pf * np.product(L) / geom.sphere_volume(R, dim)))
    pf_actual = n_to_pf(L, n, R)
    return n, pf_actual


def pack_simple(R, L, pf=None, n=None, rng=None):
    """Pack a number of non-intersecting spheres into a periodic system.

    Can specify packing by number of spheres or packing fraction.

    This implementation uses a naive uniform distribution of spheres,
    and the Tabula Rasa rule (start from scratch if an intersection occurs).

    This is likely to be very slow for high packing fractions

    Parameters
    ----------
    R: float
        Sphere radius.
    L: float array, shape (d,)
        System lengths.
    pf: float or None
        Packing fraction
    n: integer or None
        Number of spheres.
    rng: RandomState or None
        Random number generator. If None, use inbuilt numpy state.

    Returns
    -------
    r: float array, shape (n, d)
        Coordinates of the centres of the spheres for a valid configuration.
    R_actual: float
        Actual sphere radius used in the packing.
        In this implementation this will always be equal to `R`;
        it is returned only to provide a uniform interface with the
        Metropolis-Hastings implementation.
    """
    if rng is None:
        rng = np.random
    if pf is not None:
        if pf == 0.0:
            return np.array([]), R
        # If packing fraction is specified, find required number of spheres
        # and the actual packing fraction this will produce
        n, pf_actual = pf_to_n(L, pf, R)
    elif n is not None:
        if n == 0:
            return np.array([]), R

    dim = L.shape[0]
    r = np.empty([n, dim])
    while True:
        for i_dim in range(L.shape[0]):
            r[:, i_dim] = rng.uniform(-L[i_dim] / 2.0, L[i_dim] / 2.0,
                                      size=(n,))
        if not np.any(pdist_sq_periodic(r, L) < (2.0 * R) ** 2):
            return r, R


def pack(R, L, pf=None, n=None, rng=None,
         beta_max=1e4, dL_max=0.02, dr_max=0.02):
    """Pack a number of non-intersecting spheres into a periodic system.

    Can specify packing by number of spheres or packing fraction.

    This implementation uses the Metropolis-Hastings algorithm for an
    NPT system.

    Parameters
    ----------
    R: float
        Sphere radius.
    L: float array, shape (d,)
        System lengths.
    pf: float or None
        Packing fraction
    n: integer or None
        Number of spheres.
    rng: RandomState or None
        Random number generator. If None, use inbuilt numpy state.

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
    """
    if pf is not None:
        if pf == 0.0:
            return np.array([]), R
        # If packing fraction is specified, find required number of spheres
        # and the actual packing fraction this will produce
        n, pf_actual = pf_to_n(L, pf, R)
    elif n is not None:
        if n == 0:
            return np.array([]), R
        # If n is specified, find packing fraction
        pf_actual = n_to_pf(L, n, R)

    # Calculate an initial packing fraction and system size
    # Start at at most 0.5%; lower if the desired packing fraction is very low
    pf_initial = min(0.005, pf_actual / 2.0)
    # Find system size that will create this packing fraction
    dim = L.shape[0]
    increase_initial_ratio = (pf_actual / pf_initial) ** (1.0 / dim)
    L_0 = L * increase_initial_ratio

    # Pack naively into this system
    r_0, R = pack_simple(R, L_0, n=n, rng=rng)

    mg = MetroRCP(r_0, L_0, R, dr_max, dL_max, rng=rng)

    print('Initial packing done, Initial packing: {:g}'.format(mg.pf))

    t = 0
    while mg.pf() < pf_actual:
        t += 1
        beta = beta_max * mg.pf()
        mg.iterate(beta)

        if not t % every:
            print('Packing: {:.1f}%'.format(100.0 * mg.pf()))

    # print('Final packing: {:.1f}%'.format(100.0 * mg.pf()))

    return mg.r, mg.R
