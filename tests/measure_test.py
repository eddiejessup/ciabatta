from __future__ import print_function, division
import numpy as np


def measure_bcf(a):
    return a[0] / sum(a)


def measure_rss(a):
    f = [n / float(sum(a)) for n in a]
    return np.sqrt(np.sum(np.square(f)))


def measure_ss(a):
    f = [n / float(sum(a)) for n in a]
    return np.sum(np.square(f))


def measure_norm(a):
    f = [(n - 1.0) / (sum(a) - 1.0) for n in a]
    return sum(f)


def measure_weight(a):
    f = [n * (n - 1) / float(sum(a) * (sum(a) - 1)) for n in a]
    return sum(f)

measures = [measure_weight]

N = 6

tests = [
    [N],
    [N - 1, 1],
    [N - 2, 2],
    [N - 2, 1, 1],
    [N // 2, N // 2],
    [2] + [1 for _ in range(N - 1)],
    [1 for _ in range(N)],
]

for a in tests:
    for measure in measures:
        print(a, measure(a))
    print()
