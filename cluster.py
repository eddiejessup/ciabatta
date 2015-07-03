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
    if labels.min() == 0:
        return np.bincount(labels)
    elif labels.min() == 1:
        return np.bincount(labels - 1)
    else:
        raise Exception


def norm_to_colour(x):
    c = x - float(x.min())
    c /= float(x.max())
    c *= 255.0
    return c


def biggest_cluster_fraction(clust_sizes):
    """Calculate the fraction of points that lie in the biggest cluster.

    The measure to some extent indicates the degree to which points belong to
    a few clusters. However, it is a bad measure, because it gives results that
    are counter-intuitive. For example, these distributions all give the same
    result:

    - [n - 10, 10]
    - [n - 10, 5, 5]
    - [n - 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    Parameters
    ----------
    clust_sizes: list[int]
        The number of points in each cluster.

    Returns
    -------
    m: float
        Cluster measure.
    """
    clust_sizes = np.array(clust_sizes)
    return clust_sizes.max() / float(clust_sizes.sum())


def cluster_measure(clust_sizes):
    """Calculate how clumpy a set of clustered points are.

    The measure indicates the degree to which points belong to a few clusters.
    This is calculated by finding all clusters with more than one point.
    These clusters are then normalised to give their fraction of the whole.
    The square root of the sum of the square of these values then gives the
    final measure.

    This is chosen to give intuitively sensible orderings to distributions of
    points. For example, these are in order of decreasing measure,

    - [n] = 1
    - [n, 2]
    - [n, 1, 1]
    - [n / 10, 1, 1]
    - [1, 1, 1, ...] = 0

    Parameters
    ----------
    clust_sizes: list[int]
        The number of points in each cluster.

    Returns
    -------
    m: float
        Cluster measure.
    """
    clust_sizes = np.array(clust_sizes)
    nontrivial_clust_sizes = clust_sizes[clust_sizes > 1]
    nontrivial_clust_fracs = nontrivial_clust_sizes / float(clust_sizes.sum())
    return np.sqrt(np.sum(np.square(nontrivial_clust_fracs)))
