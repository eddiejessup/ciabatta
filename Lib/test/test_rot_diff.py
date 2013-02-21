import numpy as np
import utils
import matplotlib.pyplot as pp
from mpl_toolkits.mplot3d import Axes3D

def D_dependence(d=3):
    samples = 400
    Ds_calc = []
    Ds_in = np.linspace(0.00001, 1.0, num=samples)
    n = 1000
    dt = 0.1

    if d == 2: rot_diff = utils.rot_diff_2d
    elif d == 3: rot_diff = utils.rot_diff_3d

    for D_rot in Ds_in:
        a = utils.sphere_pick(d, n)
        a_rot = rot_diff(a, D_rot, dt)
        dthetas = utils.vector_angle(a, a_rot)
        D_rot_calc = (dthetas ** 2).mean() / (2.0 * dt)
        Ds_calc.append(D_rot_calc)

    pp.scatter(Ds_in, np.array(Ds_calc) / Ds_in)
    pp.show()

def diffusion_track(d):
    dt = 0.1
    D_rot = 0.01
    n = 1000
    a_rots = np.zeros([n, d])
    a_rots[0] = utils.sphere_pick(d, 1)

    if d == 2: rot_diff = utils.rot_diff_2d
    elif d == 3: rot_diff = utils.rot_diff_3d

    for i in range(1, n):
        a_rots[i] = rot_diff(np.array([a_rots[i - 1]]), D_rot, dt)

    fig = pp.figure()
    if d == 2:
        ax = fig.add_subplot(111)
        ax.scatter(a_rots[:, 0], a_rots[:, 1], c='red', s=1)
        r = 1.2
        ax.set_xlim([-r, r])
        ax.set_ylim([-r, r])
    elif d == 3:
        ax = fig.add_subplot(111, projection='3d')
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        r = 0.96
        ax.plot_surface(r*x, r*y, r*z,  rstride=4, cstride=4, color='b', shade=False)
        ax.scatter(a_rots[:, 0], a_rots[:, 1], a_rots[:, 2], c='red', s=5, facecolor='red', lw=0.0)
    ax.set_aspect('equal')
    pp.show()

if __name__ == '__main__':
    D_dependence(2)
    diffusion_track(3)
