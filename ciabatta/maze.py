"""
Algorithms related to generating and processing mazes, represented as boolean
numpy arrays.
"""
from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np


def wrap_inc(M, i):
    return i + 1 if i < M - 1 else 0


def wrap_dec(M, i):
    return i - 1 if i > 0 else M - 1


def shrink(w_old, n):
    if n < 1:
        raise Exception('Shrink factor >= 1')
    elif n == 1:
        return w_old
    elif n % 2 != 0:
        raise Exception('Shrink factor must be odd')
    M = w_old.shape[0]
    w_new = np.zeros(w_old.ndim * [M * n], dtype=w_old.dtype)
    mid = n // 2
    for x in range(M):
        x_ = x * n
        for y in range(M):
            y_ = y * n
            if w_old[x, y]:
                w_new[x_ + mid, y_ + mid] = True
                if w_old[wrap_inc(M, x), y]:
                    w_new[x_ + mid:x_ + n, y_ + mid] = True
                if w_old[wrap_dec(M, x), y]:
                    w_new[x_:x_ + mid, y_ + mid] = True
                if w_old[x, wrap_inc(M, y)]:
                    w_new[x_ + mid, y_ + mid:y_ + n] = True
                if w_old[x, wrap_dec(M, y)]:
                    w_new[x_ + mid, y * n:y_ + mid] = True
    return w_new


def make_offsets(dim):
    offsets = np.zeros([2 * dim, dim])
    offsets[:dim] = np.identity(dim)
    offsets[dim:] = -np.identity(dim)
    return offsets


def step(p, o, m, n=1):
    p_new = p + n * o
    p_new[p_new < 0] += m
    p_new[p_new > m - 1] -= m
    return p_new


def make_maze_dfs(M=27, dim=2, rng=None):
    """Generate a maze using the Depth-first search algorithm.

    http://en.wikipedia.org/wiki/Depth-first_search

    Parameters
    ----------
    M: integer.
        Size of the maze.
        The algorithm requires that the maze size is even.
    dim: integer.
        The spatial dimension.
    seed: integer or None.
        Seed for the random number generator.
        None will use a different seed for each call.

    Returns
    -------
    maze: boolean array, shape (M, M)
        Array defining the maze. Walls are represented by `True`.
    """
    if M % 2 != 0:
        raise Exception('Require Maze size to be even.')
    if rng is None:
        rng = np.random
    maze = np.zeros(dim * (M,), dtype=np.bool)
    pos = rng.randint(0, M, dim)
    maze[tuple(pos)] = True
    path = [pos]
    offsets = make_offsets(dim)
    while path:
        neighbs = []
        for offset in offsets:
            if not maze[tuple(step(pos, offset, M, 2))]:
                neighbs.append(offset)
        if neighbs:
            offset = neighbs[rng.randint(len(neighbs))]
            for i in range(2):
                pos = step(pos, offset, M)
                maze[tuple(pos)] = True
            path.append(pos)
        else:
            pos = path.pop()
    return maze


def main():
    maze = make_maze_dfs(28)
    plt.imshow(maze, interpolation='nearest')
    plt.show()

if __name__ == '__main__':
    main()
