import numpy as np
import utils
import fields
import maze as maze_module

def shrink(w_old, n):
    if n < 1: raise Exception('Shrink factor >= 1')
    if n == 1: return w_old
    if n % 2 != 0: raise Exception('Shrink factor must be odd')
    M = w_old.shape[0]
    w_new = np.zeros(w_old.ndim * [M * n], dtype=w_old.dtype)
    mid = n // 2
    for x in range(M):
        x_ = x * n
        for y in range(M):
            y_ = y * n
            if w_old[x, y]:
                w_new[x_ + mid, y_ + mid] = True
                if w_old[utils.wrap_inc(M, x), y]: 
                    w_new[x_ + mid:x_ + n, y_ + mid] = True
                if w_old[utils.wrap_dec(M, x), y]: 
                    w_new[x_:x_ + mid, y_ + mid] = True
                if w_old[x, utils.wrap_inc(M, y)]: 
                    w_new[x_ + mid, y_ + mid:y_ + n] = True
                if w_old[x, utils.wrap_dec(M, y)]: 
                    w_new[x_ + mid, y * n:y_ + mid] = True
    return w_new

class Walls(fields.Field):
    def __init__(self, dim, M, L):
        fields.Field.__init__(self, dim, M, L)
        self.a = np.zeros(self.dim * (self.M,), dtype=np.uint8)
        self.alg = 'blank'
        self.A_free_calc()

    def A_free_calc(self):
        self.A_free_i = np.logical_not(self.a).sum()
        self.A_free = (float(self.A_free_i) / float(self.A_i)) * self.A

class Closed(Walls):
    def __init__(self, dim, M, L):
        Walls.__init__(self, dim, M, L)
        self.alg = 'closed'
        self.d_i = 1
        self.d = self.d_i * self.dx
        self.a[...] = False
        for i_dim in range(self.a.ndim):
            for i_end in (0, -1):
                inds = self.a.ndim * [Ellipsis]
                inds[i_dim] = i_end
                self.a[inds] = True
        self.A_free_calc()

class TrapsN(Walls):
    def __init__(self, M, L, d, w, s, traps_f):
        if w < 0.0 or w > L:
            raise Exception('Invalid trap length')
        if s < 0.0 or s > w:
            raise Exception('Invalid slit length')

        Walls.__init__(self, 2, M, L)
        self.alg = 'trap'

        self.d_i = int(d / self.dx) + 1
        self.w_i = int(w / self.dx) + 1
        self.s_i = int(s / self.dx) + 1
        w_i_half = self.w_i // 2
        s_i_half = self.s_i // 2

        self.d = self.d_i * self.dx
        self.w = self.w_i * self.dx
        self.s = self.s_i * self.dx

        self.A_traps_i = 0
        self.traps_i = np.asarray(self.M * traps_f, dtype=np.int)
        for x, y in self.traps_i:
            self.a[x - w_i_half - self.d_i:x + w_i_half + self.d_i + 1,
                   y - w_i_half - self.d_i:y + w_i_half + self.d_i + 1] = True
            self.a[x - w_i_half:x + w_i_half + 1,
                   y - w_i_half:y + w_i_half + 1] = False
            self.a[x - s_i_half:x + s_i_half + 1,
                   y + w_i_half:y + w_i_half + self.d_i + 1] = False

        self.A_traps_i += np.logical_not(self.a[x - w_i_half:
                                                x + w_i_half + 1,
                                                y - w_i_half:
                                                y + w_i_half + 1]).sum()

        self.A_traps = (float(self.A_traps_i) / float(self.A_i)) * self.A
        self.A_free_calc()

class Traps1(TrapsN):
    def __init__(self, M, L, d, w, s):
        traps_f = np.array([[0.50, 0.50]], dtype=np.float)
        TrapsN.__init__(self, M, L, d, w, s, traps_f)

class Traps4(TrapsN):
    def __init__(self, M, L, d, w, s):
        traps_f = np.array([[0.25, 0.25], [0.25, 0.75], 
                            [0.75, 0.25], [0.75, 0.75]], dtype=np.float)
        TrapsN.__init__(self, M, L, d, w, s, traps_f)

class Traps5(TrapsN):
    def __init__(self, M, L, d, w, s):
        traps_f = np.array([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25],
                            [0.75, 0.75], [0.50, 0.50]], dtype=np.float)
        TrapsN.__init__(self, M, L, d, w, s, traps_f)

class Maze(Walls):
    def __init__(self, dim, L, d, dx, seed=None):
        if L / dx % 1 != 0:
            raise Exception('Require L / dx to be an integer')
        if L / d % 1 != 0:
            raise Exception('Require L / d to be an integer')
        if (L / dx) / (L / d) % 1 != 0:
            raise Exception('Require array size / maze size to be integer')

        Walls.__init__(self, dim, int(L / dx), L)
        self.alg = 'maze'
        self.seed = seed
        self.d = d

        self.M_m = int(self.L / self.d)
        self.d_i = int(self.M / self.M_m)
        maze = maze_module.make_maze_dfs(self.M_m, self.dim, self.seed)
        self.a[...] = utils.extend_array(maze, self.d_i)

        self.A_free_calc()
