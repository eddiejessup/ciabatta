from abc import ABCMeta, abstractmethod
import fileio
import numpy as np


class BaseModel(object):
    __metaclass__ = ABCMeta

    repr_fields = ['seed', 'dt']

    @abstractmethod
    def __init__(self, seed, dt):
        self.seed = seed
        self.dt = dt

        self.t = 0.0
        self.i = 0

        np.random.seed(self.seed)

    def iterate(self):
        self.t += self.dt
        self.i += 1

    def __repr__(self):
        return '{}_{}'.format(self.__class__.__name__,
                              fileio.reprify(self, self.repr_fields))
