import numpy as np
import utils
import field_numerics

density = field_numerics.density

class Field(object):
    def __init__(self, env, dx):
        self.env = env
        self.M = int(self.env.L / dx)
        self.dx = self.env.L / self.M

        if self.dx <= 0.0:
            raise Exception('Require space-step > 0')

    def get_A_i(self):
        return self.M ** self.env.dim

    def get_dA(self):
        return self.dx ** self.env.dim

    def r_to_i(self, r):
        return utils.r_to_i(r, self.env.L, self.dx)

    def i_to_r(self, i):
        return utils.i_to_r(i, self.env.L, self.dx)

    def iterate(self, *args, **kwargs):
        pass

class Scalar(Field):
    def __init__(self, env, dx, a_0=0.0):
        Field.__init__(self, env, dx)
        self.a = np.ones(self.env.dim * (self.M,), dtype=np.float) * a_0

    def get_grad(self):
        return field_numerics.grad(self.a, self.dx)

    def get_grad_i(self, r):
        return field_numerics.grad_i(self.a, self.r_to_i(r), self.dx)

    def get_laplacian(self):
        return field_numerics.laplace(self.a, self.dx)

class Diffusing(Scalar):
    def __init__(self, env, dx, D, a_0=0.0):
        Scalar.__init__(self, env, dx, a_0=a_0)
        self.D = D

        if self.D < 0.0:
            raise Exception('Require diffusion constant >= 0')
        if self.D > self.dx ** 2 / (2.0 * self.env.dim * self.env.dt):
            raise Exception('Unstable diffusion constant')

    def iterate(self):
        self.a += self.D * self.get_laplacian() * self.env.dt