from __future__ import print_function, division
import unittest
import numpy as np
from scipy.spatial.distance import cdist
from ciabatta import cluster


class ClusterTest(unittest.TestCase):

    def setUp(self):
        seed = 1
        self.rng = np.random.RandomState(seed)
        self.n = 100
        self.threshold = 0.11
        self.rs = self.rng.uniform(-1.0, 1.0, size=(400, 2))

    def test_flat_cluster(self):
        ls = cluster.cluster(self.rs, self.threshold)
        for i in range(self.rs.shape[0]):
            l = ls[i]
            r = self.rs[i]
            rs_clust = self.rs[np.logical_and(ls == l,
                                              np.any(r != self.rs, axis=1))]
            rs_nonclust = self.rs[ls != l]
            d_clust = cdist(np.array([r]), rs_clust)
            d_nonclust = cdist(np.array([r]), rs_nonclust)
            if d_clust.shape[1]:
                self.assertTrue(d_clust.min() < self.threshold)
                self.assertTrue(d_nonclust.min() > self.threshold)

if __name__ == '__main__':
    unittest.main()
