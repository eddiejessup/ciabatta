import numpy as np
import utils
import field_numerics

density = field_numerics.density

class Field(object):
    def __init__(self, parent_env, dx):
        self.parent_env = parent_env
        self.M = int(self.parent_env.L / dx)
        self.dx = self.parent_env.L / self.M

        if self.dx <= 0.0:
            raise Exception('Require space-step > 0')

    def get_A_i(self):
        return self.M ** self.parent_env.dim

    def get_dA(self):
        return self.dx ** self.parent_env.dim

    def r_to_i(self, r):
        return np.asarray((r + self.parent_env.L_half) / self.dx, dtype=np.int)

    def i_to_r(self, i):
        return -self.parent_env.L_half + (i + 0.5) * self.dx

    def iterate(self, *args):
        pass
    
    def output(self, dirname, prefix=''):
        pass
    
    def output_persistent(self, dirname, prefix=''):
        file = open('%s/%sparams.dat' % (dirname, prefix), 'w')
        file.write('dx,%f\n' % self.dx)
        file.close()

class Scalar(Field):
    def __init__(self, parent_env, dx, a_0=0.0):
        Field.__init__(self, parent_env, dx)
        self.a = np.ones(self.parent_env.dim * (self.M,), dtype=np.float) * a_0

    def get_grad(self):
        return field_numerics.grad(self.a, self.dx)

    def get_grad_i(self, r):
        return field_numerics.grad_i(self.a, self.r_to_i(r), self.dx)

    def get_laplacian(self):
        return field_numerics.laplace(self.a, self.dx)

    def output_persistent(self, dirname, prefix=''):
        super(Scalar, self).output_persistent(dirname, prefix)
        np.save('%s/%sa' % (dirname, prefix), self.a)

class Diffusing(Scalar):
    def __init__(self, parent_env, dx, D, a_0=0.0):
        Scalar.__init__(self, parent_env, dx, a_0=a_0)
        self.D = D

        if self.D < 0.0:
            raise Exception('Require diffusion constant >= 0')
        if self.D > self.dx ** 2 / (2.0 * self.parent_env.dim * self.parent_env.dt):
            raise Exception('Unstable diffusion constant')

    def iterate(self):
        self.a += self.D * self.get_laplacian() * self.parent_env.dt
        
    def output(self, dirname, prefix=''):
        np.save('%s/%sa' % (dirname, prefix), self.a)

    def output_persistent(self, dirname, prefix=''):
        super(Diffusing, self).output_persistent(dirname, prefix)
        file = open('%s/%sparams.dat' % (dirname, prefix), 'a')
        file.write('D,%f\n' % self.D)
        file.close()