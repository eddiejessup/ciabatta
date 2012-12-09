import numpy as np
import fields
import walls as walls_module

# Cython extension
import walled_field_numerics

class Scalar(fields.Scalar):
    def __init__(self, dim, M, L, a_0=0.0, walls=None):
        fields.Scalar.__init__(self, dim, M, L, a_0)

        if walls is None:
            self.walls = walls_module.Blank(self.dim, self.M, self.L)
        else:
            self.walls = walls

        # Make field zero-valued in walls
        self.a *= np.logical_not(self.walls.a)

    def get_grad(self):
        return walled_field_numerics.grad(self.a, self.dx, self.walls.a)

    def get_grad_i(self, inds):
        return walled_field_numerics.grad_i(self.a, inds, self.dx, self.walls.a)

    def get_laplacian(self):
        return walled_field_numerics.laplace(self.a, self.dx, self.walls.a)

# Note, inheritance order matters to get walled grad & laplacian call
# (see diamond problem on wikipedia and how python handles it)
class Diffusing(Scalar, fields.Diffusing):
    def __init__(self, dim, M, L, D, dt, a_0=0.0, walls=None):
        fields.Diffusing.__init__(self, dim, M, L, D, dt, a_0)
        Scalar.__init__(self, dim, M, L, a_0, walls)

    def iterate(self):
        laplace_a = walled_field_numerics.laplace(self.a, self.dx, self.walls.a)
        self.a += self.D * laplace_a * self.dt

class Food(Diffusing):
    def __init__(self, dim, M, L, D, dt, a_0, sink_rate, walls=None):
        Diffusing.__init__(self, dim, M, L, D, dt, a_0, walls)

        if sink_rate < 0.0:
            raise Exception('Require food sink rate >= 0')

        self.sink_rate = sink_rate

    def iterate(self, density):
        Diffusing.iterate(self)
        self.a -= self.sink_rate * density * self.dt
        self.a = np.maximum(self.a, 0.0)

class Secretion(Diffusing):
    def __init__(self, dim, M, L, D, dt, sink_rate, source_rate, walls=None):
        Diffusing.__init__(self, dim, M, L, D, dt, walls=walls)

        if source_rate < 0:
            raise Exception('Require chemo-attractant source rate >= 0')
        if sink_rate < 0:
            raise Exception('Require chemo-attractant sink rate >= 0')

        self.source_rate = source_rate
        self.sink_rate = sink_rate

    def iterate(self, density, food):
        Diffusing.iterate(self)
        self.a += (self.source_rate * density * food.a -
                   self.sink_rate * self.a) * self.dt
        self.a = np.maximum(self.a, 0.0)
