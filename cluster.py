import numpy as np
from scipy.cluster import hierarchy as hc
from scipy.spatial.distance import cdist
from _periodic_cluster import get_cluster_list


def cluster(r, r_max):
    linkage_matrix = hc.linkage(r, method='single', metric='sqeuclidean')
    return hc.fcluster(linkage_matrix, t=r_max ** 2, criterion='distance')


def cluster_periodic(r, r_max, L):
    # Get a linked list where a closed loop indicates a single cluster.
    linked_list = get_cluster_list(r, r_max, L)
    # Convert from Fortran 1-based to Python 0-based indexing.
    linked_list -= 1
    return get_labels(linked_list)


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


def get_clumpiness(clust_sizes):
    """Calculate how 'clumpy' a set of clustered points are.

    The measure indicates the degree to which points belong to a few clusters.
    This is calculated by finding the clumpiness of a point in each cluster,
    which is its contribution to the overall population. This is then
    weighted by the number of points with that clumpiness value, and the sum
    taken of this over all clusters.

    This is chosen to give intuitively sensible orderings to distributions of
    points. For example, these are examples of its values for some populations:

    - [6]: 1
    - [5, 1]: 0.67
    - [4, 2]: 0.47
    - [4, 1, 1]: 0.4
    - [3, 3]: 0.4
    - [2, 1, 1, 1, 1, 1]: 0.05
    - [1, 1, 1, 1, 1, 1]: 0

    Parameters
    ----------
    clust_sizes: list[int]
        The number of points in each cluster.

    Returns
    -------
    k: float
        Clumpiness measure.
    """
    clust_fracs = clust_sizes / float(clust_sizes.sum())
    clumpinesses = (clust_sizes - 1.0) / float(clust_sizes.sum() - 1.0)
    return np.sum(clust_fracs * clumpinesses)


def get_labels(linked_list):
    """Convert clusters represented as a linked list into a labels array.

    For example, `[1, 0, 2, 3]` represents a system where the first two samples
    are in a single cluster, because the first points to the second, then the
    second points to the first. The last two point to themselves, and are thus
    in a cluster with a single sample.

    This function converts such a list into an array of integers, whose entries
    begin at zero and increase consecutively. Each integer identifies a sample
    as belonging to a particular cluster, labelled by that integer.

    Parameters
    ----------
    linked_list: list[int]
        List of integers whose minimum must be zero,
        and maximum must be `len(linked_list)`.

    Returns
    -------
    labels: np.ndarray[dtype=int, shape=(len(linked_list),)]
        Cluster labels, starting at zero.
    """
    # `-1` indicates a sample that has not been visited yet.
    labels = np.full([len(linked_list)], -1, dtype=np.int)

    # Initial label is zero.
    label = 0
    # Each unvisited index represents a new cluster.
    for i_base in range(len(linked_list)):
        if labels[i_base] == -1:
            i = i_base
            # Iterate through the linked samples setting their label.
            while True:
                labels[i] = label
                i_next = linked_list[i]
                # When we arrive back at the start,
                # we are finished with that cluster.
                if i_next == i_base:
                    break
                i = i_next
            label += 1
    return labels
