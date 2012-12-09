import numpy as np
import field_numerics

density = field_numerics.density

class Field(object):
    def __init__(self, dim, M, L):
        if dim < 1:
            raise Exception('Require field dimension >= 1')
        if M < 1:
            raise Exception('Require field lattice size >= 1')
        if L < 0.0:
            raise Exception('Require field physical size >= 0')

        self.dim = dim
        self.M = M
        self.L = L

        self.L_half = self.L / 2.0
        self.dx = L / float(self.M)
        self.dA = self.dx ** self.dim
        self.A = self.L ** self.dim
        self.A_i = self.M ** self.dim

    def r_to_i(self, r):
        return np.asarray((r + self.L / 2.0) / self.dx, dtype=np.int)

    def i_to_r(self, i):
        return -(self.L / 2.0) + (i + 0.5) * self.dx

    def iterate(self, *args):
        pass

class Scalar(Field):
    def __init__(self, dim, M, L, a_0=0.0):
        Field.__init__(self, dim, M, L)
        self.a = np.ones(dim * (M,), dtype=np.float) * a_0
        self.dx = L / float(M)
        self.dA = self.dx ** self.dim
        self.A = self.L ** self.dim
        self.A_i = self.a.size

    def get_grad(self):
        return field_numerics.grad(self.a, self.dx)

    def get_grad_i(self, inds):
        return field_numerics.grad_i(self.a, inds, self.dx)

    def get_laplacian(self):
        return field_numerics.laplace(self.a, self.dx)

class Diffusing(Scalar):
    def __init__(self, dim, M, L, D, dt, a_0=0.0):
        Scalar.__init__(self, dim, M, L, a_0)
        if D < 0.0:
            raise Exception('Require diffusion constant >= 0')
        if dt <= 0.0:
            raise Exception('Require time-step > 0')
        self.D = D
        self.dt = dt
        self.laplace_a = np.zeros_like(self.a)

    def iterate(self):
        laplace_a = field_numerics.laplace(self.a, self.dx)
        self.a += self.D * laplace_a * self.dt
