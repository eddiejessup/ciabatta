import numpy as np
from scipy.cluster import hierarchy as hc
from scipy.spatial.distance import cdist


def test_flat_cluster():
    rs = np.random.uniform(-1.0, 1.0, size=(400, 2))
    threshold = 0.11
    ls = cluster(rs, threshold)

    for i in range(len(rs)):
        l = ls[i]
        r = rs[i]
        rs_clust = rs[np.logical_and(ls == l, np.any(r != rs, axis=1))]
        rs_nonclust = rs[ls != l]
        d_clust = cdist(np.array([r]), rs_clust)
        d_nonclust = cdist(np.array([r]), rs_nonclust)
        if d_clust.shape[1]:
            if d_clust.min() > threshold:
                print('ERROR: cluster with smallest '
                      'distance greater than threshold')
            if d_nonclust.min() < threshold:
                print('ERROR: distance between non-clustered'
                      'points less than threshold')


def cluster(r, r_max):
    return hc.fclusterdata(r, t=r_max, criterion='distance')


def nclusters(labels):
    return len(set(labels))


def cluster_sizes(labels):
    return np.bincount(labels - 1)


def norm_to_colour(x):
    c = x - float(x.min())
    c /= float(x.max())
    c *= 255.0
    return c


def biggest_cluster_fraction(labels):
    return cluster_sizes(labels).max() / float(labels.shape[0])
