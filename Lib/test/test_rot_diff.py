import numpy as np
import utils
import matplotlib.pyplot as pp
from mpl_toolkits.mplot3d import Axes3D

dt = 0.01
n = 10


def D_dependence(d=3):
    samples = 400

    Ds_calc = []
    Ds_in = np.linspace(0.00001, 1.0, num=samples)
    for D_rot in Ds_in:
        a = utils.sphere_pick(d, n)
        a_rot = utils.rot_diff(a, D_rot, dt)
        Ds_calc.append(utils.calc_D_rot(a, a_rot, dt))

    pp.scatter(Ds_in, np.array(Ds_calc) / Ds_in)
    pp.show()


def diffusion_track(d):
    D_rot = 0.01
    t = 20.0
    iters = int(round(t / dt))

    a_rots = np.zeros([iters, d])
    a_rots[0] = utils.sphere_pick(d, 1)

    for i in range(1, iters):
        a_rots[i] = utils.rot_diff(np.array([a_rots[i - 1]]), D_rot, dt)

    fig = pp.figure()
    if d == 2:
        ax = fig.add_subplot(111)
        ax.scatter(a_rots[:, 0], a_rots[:, 1], c='red', s=1)
        r = 1.2
        ax.set_xlim([-r, r])
        ax.set_ylim([-r, r])
    elif d == 3:
        ax = fig.add_subplot(111, projection='3d')
        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 60)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        r = 0.96
        ax.plot_surface(
            r * x, r * y, r * z,  rstride=4, cstride=4, color='b', shade=False)
        ax.scatter(a_rots[:, 0], a_rots[:, 1], a_rots[
                   :, 2], c='red', s=5, facecolor='red', lw=0.0)
    ax.set_aspect('equal')
    pp.show()

if __name__ == '__main__':
    D_dependence(3)
    diffusion_track(3)
