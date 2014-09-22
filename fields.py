
import numpy as np
from ciabatta import utils, field_numerics, walled_field_numerics

density = field_numerics.density


class Space(object):

    def __init__(self, L, dim):
        self.L = L
        self.L_half = L / 2.0
        self.dim = dim

    def A(self):
        return self.L ** self.dim

    def iterate(self, *args, **kwargs):
        pass


class Field(Space):

    def __init__(self, L, dim, dx):
        Space.__init__(self, L, dim)
        self.M = int(self.L / dx)

    def dx(self):
        return self.L / self.M

    def A_i(self):
        return self.M ** self.dim

    def dA(self):
        return self.dx() ** self.dim

    def r_to_i(self, r):
        return utils.r_to_i(r, self.L, self.dx())

    def i_to_r(self, i):
        return utils.i_to_r(i, self.L, self.dx())


class Scalar(Field):

    def __init__(self, L, dim, dx, a_0=0.0):
        Field.__init__(self, L, dim, dx)
        self.a = np.ones(self.dim * (self.M,), dtype=np.float) * a_0

    def grad(self):
        return field_numerics.grad(self.a, self.dx())

    def grad_i(self, r):
        return field_numerics.grad_i(self.a, self.r_to_i(r), self.dx())

    def laplacian(self):
        return field_numerics.laplace(self.a, self.dx())


class Diffusing(Scalar):

    def __init__(self, L, dim, dx, D, dt, a_0=0.0):
        Scalar.__init__(self, L, dim, dx, a_0=a_0)
        self.D = D
        self.dt = dt

        if self.D > self.dx() ** 2 / (2.0 * self.dim * self.dt):
            raise Exception('Unstable diffusion constant')

    def iterate(self):
        self.a += self.D * self.laplacian() * self.dt


class WalledScalar(Scalar):
    def __init__(self, L, dim, dx, obstructs, a_0=0.0):
        Scalar.__init__(self, L, dim, dx, a_0=a_0)
        # Make field zero-valued where obstructed
        self.of = obstructs.to_field(self.dx())
        self.a *= np.logical_not(self.of)

    def grad(self):
        return walled_field_numerics.grad(self.a, self.dx(), self.of)

    def grad_i(self, r):
        return walled_field_numerics.grad_i(self.a, self.r_to_i(r), self.dx(),
                                            self.of)

    def laplacian(self):
        return walled_field_numerics.laplace(self.a, self.dx(), self.of)
