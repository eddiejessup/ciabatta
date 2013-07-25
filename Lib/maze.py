'''
Algorithms related to generating and processing mazes, represented as boolean
numpy arrays.
'''

from __future__ import print_function
import numpy as np
import utils

def shrink(w_old, n):
    if n < 1: raise Exception('Shrink factor >= 1')
    elif n == 1: return w_old
    elif n % 2 != 0: raise Exception('Shrink factor must be odd')
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

def step(p, o, m, n=1):
    p_new = np.array(p) + n * np.array(o)
    p_new[p_new < 0] += m
    p_new[p_new > m - 1] -= m
    return tuple(p_new)

def make_offsets(dim):
    offsets = []
    for d in range(dim):
        for sign in (-1, +1):
            offset = dim * [0]
            offset[d] = sign
            offsets.append(offset)
    return tuple(offsets)

def make_maze_dfs(M=27, dim=2, seed=None):
    ''' Generate a maze using the depth first search algorithm '''
    if M <= 1:
        raise Exception('Require Maze size > 1.')
    if M % 2 != 0:
        raise Exception('Require Maze size to be even.')
    rng = np.random.RandomState(seed)
    maze = np.zeros(dim * (M,), dtype=np.bool)
    pos = tuple(rng.randint(0, M, dim))
    maze[pos] = True
    path = [pos]
    offsets = make_offsets(dim)
    while len(path) > 0:
        neighbs = []
        for offset in offsets:
            if not maze[step(pos, offset, M, 2)]:
                neighbs.append(offset)
        if len(neighbs) > 0:
            offset = neighbs[rng.randint(len(neighbs))]
            pos = step(pos, offset, M)
            maze[pos] = True
            pos = step(pos, offset, M)
            maze[pos] = True
            path.append(pos)
        else:
            pos = path.pop()
    return maze

def main():
    maze = make_maze_dfs(30)
    import matplotlib.pyplot as pp
    pp.imshow(maze, interpolation='nearest')
    pp.show()
    pp.savefig('maze.png')

if __name__ == '__main__': main()
