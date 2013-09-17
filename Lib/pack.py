import numpy as np
import matplotlib.pyplot as pp
from matplotlib.patches import Circle, PathPatch
from mpl_toolkits.mplot3d import Axes3D 
import mpl_toolkits.mplot3d.art3d as art3d
import utils
import geom

class MetroRCP(object):
    def __init__(self, r_0, V_0, R, dr_max, dV_max):
        self.r = r_0
        self.n, self.dim = self.r.shape
        self.L = V_0 ** (1.0 / self.dim)
        self.R = R
        self.D_sq = (2.0 * self.R) ** 2
        self.dr_max = dr_max
        self.dV_max = dV_max

        self.V_full = self.n * geom.sphere_volume(self.R, self.dim)

        self.r_diff_sq = utils.vector_mag_sq(self.r[np.newaxis, :] - self.r[:, np.newaxis])
        self.r_diff_sq[np.identity(self.n, dtype=np.bool)] = np.inf

    def U(self):
        if np.any(np.abs(self.r) > self.L/2.0 - self.R): return np.inf
        if np.any(self.r_diff_sq < self.D_sq): return np.inf
        return 1.0 / self.pf()

    def displace_r(self):
        self.i = np.random.randint(self.n)
        self.r_old = self.r[self.i].copy()

        self.r[self.i] += np.random.uniform(-self.dr_max * self.L, self.dr_max * self.L, self.dim)

        self.r_diff_sq[self.i, :] = self.r_diff_sq[:, self.i] = utils.vector_mag_sq(self.r[self.i] - self.r)
        self.r_diff_sq[self.i, self.i] = np.inf

    def revert_r(self):
        self.r[self.i] = self.r_old.copy()

        self.r_diff_sq[self.i, :] = self.r_diff_sq[:, self.i] = utils.vector_mag_sq(self.r[self.i] - self.r)
        self.r_diff_sq[self.i, self.i] = np.inf

    def displace_V(self):
        self.L_old = self.L

        V_new = self.V() * (1.0 + np.random.uniform(-self.dV_max, self.dV_max))
        self.L = V_new ** (1.0 / self.dim)
        self.r *= self.L / self.L_old

        self.r_diff_sq *= (self.L / self.L_old) ** 2

    def revert_V(self):
        self.r /= self.L / self.L_old

        self.r_diff_sq /= (self.L / self.L_old) ** 2

        self.L = self.L_old

    def iterate(self, beta):
        U_0 = self.U()

        i = np.random.randint(self.n + 1)
        if i < len(self.r):
            self.displace_r()
            revert = self.revert_r
        else:
            self.displace_V()
            revert = self.revert_V

        U_new = self.U()
        if np.exp(-beta * (U_new - U_0)) < np.random.uniform():
            revert()

    def V(self):
        return self.L ** self.dim

    def pf(self):
        return self.V_full / self.V()

def random_base(n, V, dim, R):
    L = V ** (1.0 / dim)
    r = np.zeros([n, dim], dtype=np.float)
    lim = L / 2.0 - R
    for i in range(n):
        print('Initial packing: %.1f%% done' % ((i*100.0)/n))
        while True:
            r[i] = np.random.uniform(-lim, lim, dim)
            r_diff_sq = utils.vector_mag_sq(r[:i+1, np.newaxis] - r[np.newaxis, :i+1])
            if i == 0 or r_diff_sq[r_diff_sq > 0.0].min() > (2.0 * R) ** 2: break
    return r

def random_simple(pf, dim, R):
    n = int(round(pf / geom.sphere_volume(R, dim)))
    return random_base(n, dim, 1.0, R)

every = 5000

def random(pf, dim, R, beta_max=1e4, V_0=2.0, dV_max=0.02, dr_max=0.02, vis=False):
    n = int(round(pf / geom.sphere_volume(R, dim)))
    r_0 = random_base(n, V_0, dim, R)
    mg = MetroRCP(r_0, V_0, R, dr_max, dV_max)

    if vis:
        box = vp.box(pos=mg.dim*(0.0,), length=mg.L, height=mg.L, width=mg.L, opacity=0.5)
        spheres = []
        for r in mg.r:
            spheres.append(vp.sphere(pos=r, radius=mg.R, color=vp.color.blue))

    t = 0
    while mg.pf() < pf:
        t += 1
        beta = beta_max * mg.pf()
        mg.iterate(beta)

        if not t % every:
            print('Packing: %.1f%%' % (100.0*mg.pf()))

        if vis:
            box.size = mg.dim * (mg.L,)
            for sphere, r in zip(spheres, mg.r):
                sphere.pos = r
            vp.rate(every)

    return mg.r/mg.L, mg.R/mg.L

def save(r, R, fname='out.npz'):
    np.savez(fname, r=r, R=R)

def sphere_plot(r, R):
    n, dim = r.shape
    fig = pp.figure()

    if dim == 2:
        ax = fig.gca()

        for r in r:
            ax.add_artist(pp.Circle(r, R))
    else:
        ax = fig.gca(projection='3d')

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = R * np.outer(np.cos(u), np.sin(v))
        y = R * np.outer(np.sin(u), np.sin(v))
        z = R * np.outer(np.ones(np.size(u)), np.cos(v))
        for r in r:
            # ax.plot_wireframe(r[0] + x, r[1] + y, r[2] + z, color="r")
            ax.plot_surface(r[0] + x, r[1] + y, r[2] + z,  rstride=15, cstride=15)

        ax.set_zlim3d(-0.5, 0.5)

    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_aspect('equal')
    pp.show()

if __name__ == '__main__':
    import visual as vp
    random(0.4, 3, 0.1, vis=True)
