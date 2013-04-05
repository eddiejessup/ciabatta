'''
Algorithms related to generating and processing mazes, represented as boolean
numpy arrays.
'''

from __future__ import print_function
import numpy as np
import utils

def step(p, o, m, n=1):
    p_new = np.array(p) + n * np.array(o)
    p_new[p_new < 0] += m
    p_new[p_new > m - 1] -= m
    return tuple(p_new)

def make_offsets(dim):
    offsets = []
    for d in range(dim):
        for sign in (-1, +1):
            offset = [0, 0]
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
