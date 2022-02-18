from collections import defaultdict
import numpy as np

def init_huang(X, n_clusters, dissim, random_state):
    """Initialize centroids according to method by Huang [1997].

    :param X:           Numpy array representing the current snapshot of the data
    :param n_clusters:  Number of clusters to calculate.
    :param dissim:      Function to calculate the dissimilarity between datapoints
    :random_state:      RandomState object to create random numbers

    :returns initialized centroids
    """
    if X.size==0:
        return np.empty(0)

    n_attrs = X.shape[1]
    centroids = np.empty((n_clusters, n_attrs), dtype='object')
    # determine frequencies of attributes
    for iattr in range(n_attrs):
        # Sample centroids using the probabilities of attributes.
        # Note: sampling using population in static list with as many choices
        # as frequency counts. Since the counts are small integers,
        # memory consumption is low.
        choices = X[:, iattr]
        # So that we are consistent between Python versions,
        # each with different dict ordering.
        choices = sorted(choices)
        centroids[:, iattr] = random_state.choice(choices, n_clusters)
    # The previously chosen centroids could result in empty clusters,
    # so set centroid to closest point in X.
    for ik in range(n_clusters):
        ndx = np.argsort(dissim(X, centroids[ik]))
        # We want the centroid to be unique, if possible.
        while np.all(X[ndx[0]] == centroids, axis=1).any() and ndx.shape[0] > 1:
            ndx = np.delete(ndx, 0)
        centroids[ik] = X[ndx[0]]

    return centroids

def init_cao(X, n_clusters, dissim, **kwargs):
    """Initialize centroids according to method by Cao et al. [2009].
    Note: O(N * attr * n_clusters**2), so watch out with large n_clusters

    :param X:           Numpy array representing the current snapshot of the data
    :param n_clusters:  Number of clusters to calculate.
    :param dissim:      Function to calculate the dissimilarity between datapoints

    :returns initialized centroids
    """
    if X.size==0:
        return np.empty(0)
    n_points, n_attrs = X.shape
    centroids = np.empty((n_clusters, n_attrs), dtype='object')
    # Method is based on determining density of points.
    dens = np.zeros(n_points)
    for iattr in range(n_attrs):
        freq = defaultdict(int)
        for val in X[:, iattr]:
            freq[val] += 1
        for ipoint in range(n_points):
            dens[ipoint] += freq[X[ipoint, iattr]] 
    dens /= (float(n_points) * float(n_attrs))


    # Choose initial centroids based on distance and density.
    centroids[0] = X[np.argmax(dens)]
    if n_clusters > 1:
        # For the remaining centroids, choose maximum dens * dissim to the
        # (already assigned) centroid with the lowest dens * dissim.
        for ik in range(1, n_clusters):
            dd = np.empty((ik, n_points))
            for ikk in range(ik):
                dd[ikk] = dissim(X, centroids[ikk], **kwargs) * dens
            centroids[ik] = X[np.argmax(np.min(dd, axis=0))]

    return centroids

def init_cao_lists(X, n_clusters, dissim):
    """Initialize centroids according to method by Cao et al. [2009], adjusted for collection types.

    Note: O(N * attr * n_clusters**2), so watch out with large n_clusters

    :param X:           Numpy array representing the current snapshot of the data
    :param n_clusters:  Number of clusters to calculate.
    :param dissim:      Function to calculate the dissimilarity between datapoints

    :returns initialized centroids
    """
    if X.size==0:
        return np.empty(0)
    n_points, n_attrs = X.shape
    centroids = np.empty((n_clusters, n_attrs), dtype='object')
    # Method is based on determining density of points.
    dens = np.zeros(n_points)
    for iattr in range(n_attrs):

        freq = defaultdict(np.int32)
        for value_list in X[:, iattr]:
            transformed_value_list = str(value_list)
            freq[transformed_value_list] += 1
        for ipoint in range(n_points):
            dens[ipoint] += freq[str(X[ipoint, iattr])]
    dens /= (float(n_points) * float(n_attrs))

    # Choose initial centroids based on distance and density.
    centroids[0] = X[np.argmax(dens)]
    if n_clusters > 1:
        # For the remaining centroids, choose maximum dens * dissim to the
        # (already assigned) centroid with the lowest dens * dissim.
        for ik in range(1, n_clusters):
            dd = np.empty((ik, n_points))
            for ikk in range(ik):
                dd[ikk] = dissim(X, centroids[ikk]) * dens
            centroids[ik] = X[np.argmax(np.min(dd, axis=0))]

    return centroids