"""
K-prototypes clustering for mixed categorical and numerical data

Implementation copied and enhanced based on this repository: https://github.com/nicodv/kmodes
"""

from collections import defaultdict

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array
from tqdm import tqdm
from copy import deepcopy
import logging
from . import kmodes
from .util import get_max_value_key, pandas_to_numpy, find_max_frequency_attribute
from .util.dissim import matching_dissim, euclidean_dissim, matching_dissim_lists, time_dissim
from .util.init_methods import init_cao, init_cao_lists, init_huang
from src.utils import _split_num_cat, initialize_gamma, WouldTryRandomInitialization
from config import K_PROTOTYPE_REPEAT_NUM, UPDATE_GAMMA_EACH_ITERATION

# Number of tries we give the initialization methods to find non-empty
# clusters before we switch to random initialization.
MAX_INIT_TRIES = 20

class KPrototypes(kmodes.KModes):
    """k-protoypes clustering algorithm for mixed numerical/categorical data.

    Parameters
    -----------
    :param n_clusters:          Number of clusters to calculate.
    :param max_iter:            Maximum number of iterations of the k-modes algorithm for a single run.
    :param num_dissim:          Dissimilarity function used by the algorithm for numerical variables.
                                Defaults to the Euclidian dissimilarity function.
    :param cat_dissim:          Dissimilarity function used by the kmodes algorithm for categorical variables.
                                Defaults to the matching dissimilarity function.
    :param n_init:              Number of time the k-modes algorithm will be run with different
                                centroid seeds. The final results will be the best output of
                                n_init consecutive runs in terms of cost.
    :param init:                {'Huang', 'Cao'}, default: 'Cao'
                                Method for initialization:
                                'Huang': Method in Huang [1997, 1998]
                                'Cao': Method in Cao et al. [2009]
    :param gamma:               Weighing factor that determines relative importance of numerical vs.
                                other attributes (see discussion in Huang [1997]). By default,
                                automatically calculated from data.
    :param verbose:             Verbosity mode.
    :param random_state:        RandomState instance or None, optional, default: None
                                If int, random_state is the seed used by the random number generator;
                                If RandomState instance, random_state is the random number generator;
                                If None, the random number generator is the RandomState instance used
                                by `np.random`.
    :param n_jobs:              The number of jobs to use for the computation. This works by computing
                                each of the n_init runs in parallel.
                                If -1 all CPUs are used. If 1 is given, no parallel computing code is
                                used at all, which is useful for debugging. For n_jobs below -1,
                                (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
                                are used.
    :param initial_centroids    Given initial centroids. Uses the numerical part to not have a random initialization.

    Notes
    -----
    See:
    Huang, Z.: Extensions to the k-modes algorithm for clustering large
    data sets with categorical values, Data Mining and Knowledge
    Discovery 2(3), 1998.

    """

    def __init__(self, n_clusters=-1, max_iter=100, num_dissim=euclidean_dissim,
                 cat_dissim=matching_dissim, list_dissim=matching_dissim_lists, time_dissim=time_dissim, init='Cao', n_init=K_PROTOTYPE_REPEAT_NUM, gamma=None,
                 verbose=False, random_state=None, n_jobs=1, initial_centroids=None):

        super(KPrototypes, self).__init__(n_clusters, max_iter, cat_dissim, init,
                                          verbose=verbose, random_state=random_state,
                                          n_jobs=n_jobs)
        self.num_dissim = num_dissim
        self.list_dissim = list_dissim
        self.time_dissim = time_dissim
        self.gamma = gamma
        self.n_init = n_init
        self.newly_initialized_centroids=initial_centroids
        if isinstance(self.init, list) and self.n_init > 1:
            if self.verbose:
                print("Initialization method is deterministic. "
                      "Setting n_init to 1.")
            self.n_init = 1

    def fit(self, X, indices_map=None):
        """Compute k-prototypes clustering.

        :param X:               array-like, shape=[n_samples, n_features]
        :param indices_map:     Index of columns that contain numerical, categorical, list and time data

        """
        numerical=indices_map["num_indices"]
        categorical=indices_map["cat_indices"]
        list_indices=indices_map["list_indices"]
        time_indices=indices_map["time_indices"]

        time_max_values = indices_map["time_max_values"]
        if categorical is not None:
            assert isinstance(categorical, (int, list, tuple)), "The 'categorical' \
                argument needs to be an integer with the index of the categorical \
                column in your data, or a list or tuple of several of them, \
                but it is a {}.".format(type(categorical))

        if list_indices is not None:
            assert isinstance(list_indices, (int, list, tuple)), "The 'list_indices' \
                argument needs to be an integer with the index of the categorical \
                column in your data, or a list or tuple of several of them, \
                but it is a {}.".format(type(list_indices))

        if time_indices is not None:
            assert isinstance(time_indices, (int, list, tuple)), "The 'time_indices' \
                argument needs to be an integer with the index of the categorical \
                column in your data, or a list or tuple of several of them, \
                but it is a {}.".format(type(time_indices))

 

        X = pandas_to_numpy(X)

        random_state = check_random_state(self.random_state)
        # If self.gamma is None, gamma will be automatically determined from
        # the data. The function below returns its value.
        self._enc_cluster_centroids, self.labels_, self.cost_, \
        self.n_iter_, self.epoch_costs_, self.gamma, self.newly_initialized_centroids = adjusted_k_prototypes(
            X,
            numerical,
            categorical,
            list_indices,
            time_indices,
            time_max_values,
            self.n_clusters,
            self.max_iter,
            self.num_dissim,
            self.cat_dissim,
            self.list_dissim,
            self.time_dissim,
            self.gamma,
            self.init,
            self.n_init,
            self.verbose,
            random_state,
            self.n_jobs,
            self.newly_initialized_centroids
        )

        logging.info(f"\nK-Prototypes run is done. Costs at each epoch: : {self.epoch_costs_}")

        return self

    def predict(self, X, indices_map=None):
        """Predict the closest cluster each sample in X belongs to.

        :param X:               array-like, shape = [n_samples, n_features]
        :param indices_map:     Index of columns that contain numerical, categorical, list and time data

        :returns Index of the cluster each sample belongs to, Costs and the CH-Index (to assert cluster quality)
        """
        assert hasattr(self, '_enc_cluster_centroids'), "Model not yet fitted."
        numerical=indices_map["num_indices"]
        categorical=indices_map["cat_indices"]
        list_indices=indices_map["list_indices"]
        time_indices=indices_map["time_indices"]
        time_max_values = indices_map["time_max_values"]
        if categorical is not None:
            assert isinstance(categorical, (int, list, tuple)), "The 'categorical' \
                argument needs to be an integer with the index of the categorical \
                column in your data, or a list or tuple of several of them, \
                but it is a {}.".format(type(categorical))

        X = pandas_to_numpy(X)
        Xnum, Xcat, Xlist, Xtime = _split_num_cat(X, numerical, categorical, list_indices, time_indices)
        if Xnum.size > 0:
            Xnum = check_array(Xnum)

        if Xcat.size >0 :
            Xcat = check_array(Xcat, dtype=None)
        #Can't check list_indices since it contains nested lists of np.objects.
        if Xtime.size > 0:
            Xtime = check_array(Xtime)
        n_points = X.shape[0]
        
        labels, cost = labels_cost(Xnum, Xcat, Xlist, Xtime, self._enc_cluster_centroids,
                    self.num_dissim, self.cat_dissim, self.list_dissim, self.time_dissim, time_max_values, self.gamma, n_points, self.verbose)

        ch_index = adjusted_ch_index(Xnum, Xcat, Xlist, Xtime, self._enc_cluster_centroids,
                           self.num_dissim, self.cat_dissim, self.list_dissim, self.time_dissim, time_max_values, self.gamma,
                           self.labels_, cost, self.n_clusters)

        self.ch_index = ch_index
        return labels, cost, ch_index

    @property
    def cluster_centroids_(self):
        if hasattr(self, '_enc_cluster_centroids'):
            return self._enc_cluster_centroids
        else:
            raise AttributeError("'{}' object has no attribute 'cluster_centroids_' "
                                 "because the model is not yet fitted.")

def adjusted_k_prototypes(X, numerical_indices, categorical_indices, list_indices, time_indices, time_max_values, n_clusters, max_iter, num_dissim, cat_dissim, list_dissim, time_dissim,
                 gamma, init, n_init, verbose, random_state, n_jobs, cluster_centroids=None):
    """ Adjusted k-Prototypes algorithm

    :param X:                                                                       The main dataset containing all different features
    :param numerical_indices, categorical_indices, list_indices, time_indices:      Indices in dataset X to associate different parts
    :param time_max_values                                                          Max Values for the Time Indices
    :param n_clusters:                                                              Number of clusters to calculate
    :param max_iter:                                                                Maximum iteration number for clustering (afterwards process will stop)
    :param num_dissim, cat_dissim, list_dissim, time_dissim:                        Different distance functions
    :param gamma:                                                                   Weighting factor for non-numerical values
    :param init:                                                                    Cluster initiation method (e.g. 'Huang', 'Cao', 'Random')
    :param n_init:                                                                  Number of repetitions for the given config
    :param verbose:                                                                 Verbosity mode.
    :param random_state:                                                            Random state for random number generation
    :param n_jobs:                                                                  Number of threads to use (each k-prototype calculation is single-threaded, but you can calc multiple at the same time)
    :returns best output from k-Prototypes, e.g. centroids, costs, labels, all_costs
    
    """
    random_state = check_random_state(random_state)
    if sparse.issparse(X):
        raise TypeError("k-prototypes does not support sparse data.")

    if categorical_indices is None or not categorical_indices:
        raise NotImplementedError(
            "No categorical data selected, effectively doing k-means. "
            "Present a list of categorical columns, or use scikit-learn's "
            "KMeans instead."
        )
    if isinstance(categorical_indices, int):
        categorical_indices = [categorical_indices]
    assert len(categorical_indices) != X.shape[1], \
        "All columns are categorical, use k-modes instead of k-prototypes."
    assert max(categorical_indices) < X.shape[1], \
        f"Categorical index larger than number of columns: {max(categorical_indices)} vs {X.shape[1]}."

    ncatattrs = len(categorical_indices)
    nlistattrs = len(list_indices)
    ntimeattrs = len(time_indices)
    nnumattrs = len(numerical_indices)
    n_points = X.shape[0]
    assert n_clusters <= n_points, "Cannot have more clusters ({}) " \
                                   "than data points ({}).".format(n_clusters, n_points)

    Xnum, Xcat, Xlist, Xtime = _split_num_cat(X, numerical_indices, categorical_indices, list_indices, time_indices)
    
    if Xnum.size > 0:
        Xnum = check_array(Xnum)

    if Xcat.size >0 :
        Xcat = check_array(Xcat, dtype=None)
    #Can't check list_indices since it contains nested lists of np.objects.
    if Xtime.size > 0:
        Xtime = check_array(Xtime)

    # Estimate a good value for gamma, which determines the weighing of
    # categorical values in clusters (see Huang [1997]).
    if gamma is None:
        gamma = initialize_gamma(Xnum, n_clusters)

    results = []
    seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
    if n_jobs == 1:
        for init_no in range(n_init):
            results.append(_k_prototypes_single(Xnum, Xcat, Xlist, Xtime, nnumattrs, ncatattrs, nlistattrs, ntimeattrs,
                                                n_clusters, n_points, max_iter,
                                                num_dissim, cat_dissim, list_dissim, time_dissim, time_max_values, gamma,
                                                init, init_no, verbose, seeds[init_no], cluster_centroids))
    else:
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_k_prototypes_single)(Xnum, Xcat, Xlist, Xtime, nnumattrs, ncatattrs, nlistattrs, ntimeattrs,
                                          n_clusters, n_points, max_iter,
                                          num_dissim, cat_dissim, list_dissim, time_dissim, time_max_values, gamma,
                                          init, init_no, verbose, seed, cluster_centroids)
            for init_no, seed in enumerate(seeds))
    all_centroids, all_labels, all_costs, all_n_iters, all_epoch_costs, all_initialized_clusters = zip(*results)

    best = np.argmin(all_costs)
    if n_init > 1 and verbose:
        print("Best run was number {}".format(best + 1))

    return all_centroids[best], all_labels[best], all_costs[best], \
        all_n_iters[best], all_epoch_costs[best], gamma, all_initialized_clusters[best]


def _k_prototypes_single(Xnum, Xcat, Xlist, Xtime, nnumattrs, ncatattrs, nlistattrs, ntimeattrs, n_clusters, n_points,
                         max_iter, num_dissim, cat_dissim, list_dissim, time_dissim, time_max_values, gamma, init, init_no,
                         verbose, random_state, initial_cluster_centroids):
    """ Adjusted k-Prototypes algorithm

    :param Xnum, Xcat, Xlist, Xtime:                            Different parts of the dataset
    :param centroids:                                           Current centroids for k-Prototypes
    :param nnumattrs, ncatattrs, nlistattrs, ntimeattrs:        Different number of features
    :param n_clusters:                                          Number of clusters to calculate 
    :param n_points:                                            Number of datapoints
    :param max_iter:                                            Maximum iteration number for clustering (afterwards process will stop)
    :param num_dissim, cat_dissim, list_dissim, time_dissim:    Different distance functions
    :param time_max_values
    :param gamma:                                               Weighting factor for non-numerical values
    :param init:                                                Cluster initiation method (e.g. 'Huang', 'Cao', 'Random')
    :param init_no:                                             Current repetition number
    :param verbose:                                             Verbosity mode.
    :param random_state:                                        Random state for random number generation

    :returns current output from k-Prototypes, e.g. centroids, costs, labels, all_costs, initial centroids
    
    """
    
    # For numerical part of initialization, we don't have a guarantee
    # that there is not an empty cluster, so we need to retry until
    # there is none.
    random_state = check_random_state(random_state)
    init_tries = 0
    logging.info(f"Starting a cluster run with {n_clusters} clusters..")
    previously_tried_initializing = False
    newly_initialized_cluster = None
    while True:
        init_tries += 1
        # _____ INIT _____
    
        if verbose:
            print("Init: initializing centroids")
        if isinstance(init, str) and init.lower() == 'huang':
            centroids_categorical = init_huang(Xcat, n_clusters, cat_dissim, random_state)
            centroids_lists = init_huang(Xlist, n_clusters, list_dissim, random_state)
            centroids_time = init_huang(Xtime, n_clusters, time_dissim, random_state)
            #centroids_categorical = init_huang(Xcat, n_clusters, cat_dissim, random_state)
        elif isinstance(init, str) and init.lower() == 'cao':
            centroids_categorical = init_cao(Xcat, n_clusters, cat_dissim)
            centroids_lists = init_cao_lists(Xlist, n_clusters, list_dissim)
            centroids_time = init_cao(Xtime, n_clusters, time_dissim, time_max_values=time_max_values)
        elif isinstance(init, str) and init.lower() == 'random':
            raise WouldTryRandomInitialization("Random method is not a good idea for big-data..")
        elif isinstance(init, list):
            raise NotImplementedError("List method is not yet implemented for lists and time")
        else:
            raise NotImplementedError("Initialization method not supported.")

        if not initial_cluster_centroids:
            # Numerical is initialized by drawing from normal distribution,
            meanx = np.mean(Xnum, axis=0) if Xnum.size!=0 else 0
            stdx = np.std(Xnum, axis=0) if Xnum.size!=0 else 0
            centroids_numerical = meanx + random_state.randn(n_clusters, nnumattrs) * stdx,

        # Only Use numerical part for initialization (No Randomness, but fairness for new feature-set, since Cao applies weights depending on dimensions) 
        else:
            centroids_numerical = initial_cluster_centroids

        centroids = [
            centroids_numerical[0],
            centroids_categorical,
            centroids_lists,
            centroids_time
        ]

        newly_initialized_cluster=deepcopy(centroids)

        # From here centroids is an array, where entry1 represents the numerical mean+std and entry2 represents the old centroids
        if verbose:
            print("Init: initializing clusters")
        membship = np.zeros((n_clusters, n_points), dtype=np.uint8)

        # Same for the membership sum per cluster
        cl_memb_sum = np.zeros(n_clusters, dtype=int)

        # Keep track of the sum of attribute values per cluster so that we
        # can do k-means on the numerical attributes.

        cl_attr_sum_num = np.zeros((n_clusters, nnumattrs), dtype=np.float64)
        # cl_attr_freq is a list of lists with dictionaries that contain
        # the frequencies of values per cluster and attribute.
        cl_attr_freq_cat = [[defaultdict(int) for _ in range(ncatattrs)]
                        for _ in range(n_clusters)]

        cl_attr_freq_list = [[defaultdict(int) for _ in range(nlistattrs)]
                        for _ in range(n_clusters)]

        cl_attr_freq_time = [[defaultdict(int) for _ in range(ntimeattrs)]
                        for _ in range(n_clusters)]
        
        if verbose: 
            print("Initial Cluster Setup")
        for ipoint in tqdm(range(n_points), mininterval=.5, disable= not verbose):
            # Initial assignment to clusters
            # gamma[0]: since gamma is equal on every cluster initially, just arbitrarely choose index 0.
            num_distances=num_dissim(centroids[0], Xnum[ipoint]) if Xnum.size!=0 else 0
            cat_distances=cat_dissim(centroids[1], Xcat[ipoint], X=Xcat, membship=membship) if Xcat.size!=0 else 0
            list_distances=list_dissim(centroids[2], Xlist[ipoint]) if Xlist.size!=0 else 0
            time_distances=time_dissim(centroids[3], Xtime[ipoint], time_max_values=time_max_values) if Xtime.size!=0 else 0
            clust = np.argmin(num_distances + gamma * (cat_distances + list_distances + time_distances))
            
            membship[clust, ipoint] = 1
            # How many datapoints are in the current cluster!
            cl_memb_sum[clust] += 1
            # Count attribute values per cluster.

            # For each numerical feature in the current datapoint: add attribute value to the matrix!
            if Xnum.size!=0:
                for iattr, curattr in enumerate(Xnum[ipoint]):
                    cl_attr_sum_num[clust, iattr] += curattr
            # For each categorical feature in the current datapoint: add counter to the current attribute!
            if Xcat.size!=0:
                for iattr, curattr in enumerate(Xcat[ipoint]):
                    cl_attr_freq_cat[clust][iattr][curattr] += 1
            if Xlist.size!=0:   
                for iattr, curattr_list in enumerate(Xlist[ipoint]):
                    # Transform cur_attr_list to str, to count frequency of occurence
                    cl_attr_freq_list[clust][iattr][str(curattr_list)] +=1
            # For each time feature in the current datapoint: add counter to the current attribute!        
            if Xtime.size!=0:
                for iattr, curattr in enumerate(Xtime[ipoint]):
                    cl_attr_freq_time[clust][iattr][curattr] += 1
            
        # If no empty clusters, then consider initialization finalized.
        if membship.sum(axis=1).min() > 0:
            break

        if init_tries == MAX_INIT_TRIES:
            # Could not get rid of empty clusters. Randomly
            # initialize instead.
            raise RuntimeError(f"Tried to initialize clusters more than {MAX_INIT_TRIES} times unsuccessfully. Quitting...")

    # Perform an initial centroid update.
    for ik in range(n_clusters):
        for iattr in range(nnumattrs):
            # I think this basically averages the centroid value to the avg member attribute value
            centroids[0][ik, iattr] = cl_attr_sum_num[ik, iattr] / cl_memb_sum[ik]
        for iattr in range(ncatattrs):
            # Argument are the frequencies of the unique values for this attribute in the cluster
            # Returns the maximum category in the given cluster (most frequent category builds the new cluster centroid)
            centroids[1][ik, iattr] = get_max_value_key(cl_attr_freq_cat[ik][iattr])
            
        for iattr in range(nlistattrs):
            centroids[2][ik, iattr] = eval(get_max_value_key(cl_attr_freq_list[ik][iattr]))

        for iattr in range(ntimeattrs):
            centroids[3][ik, iattr] = get_max_value_key(cl_attr_freq_time[ik][iattr])

    # _____ ITERATION _____
    if verbose:
        print("Starting iterations...")
    itr = 0
    labels = None
    converged = False
    cost = costs(Xnum, Xcat, Xlist, Xtime, centroids,
                          num_dissim, cat_dissim, list_dissim, time_dissim, time_max_values, gamma, membship, n_clusters=n_clusters)

    epoch_costs = [cost]
    while itr < max_iter and not converged:
        itr += 1

        if UPDATE_GAMMA_EACH_ITERATION:
            gamma = _update_gamma_values(gamma, Xnum, membship, n_clusters)
            logging.info(f"\nUpdated Gamma Values: {gamma}")

        centroids, moves, membship = _k_prototypes_iter(Xnum, Xcat, Xlist, Xtime, centroids,
                                              cl_attr_sum_num, cl_memb_sum, cl_attr_freq_cat, cl_attr_freq_list, cl_attr_freq_time,
                                              membship, num_dissim, cat_dissim, list_dissim, time_dissim, time_max_values, gamma,
                                              random_state, verbose, n_points)
        # All points seen in this iteration
        labels, ncost = labels_cost(Xnum, Xcat, Xlist, Xtime, centroids,
                                    num_dissim, cat_dissim, list_dissim, time_dissim, time_max_values, gamma, n_points, verbose, membship)

        # INFO: Optional "early stopping" here when the cost are only 99.99% instead of strictly larger or less than 10 points have been moved
        converged = (moves == 0) or (ncost >= cost)
        epoch_costs.append(ncost)
        cost = ncost
        if verbose:
            print("Run: {}, iteration: {}/{}, moves: {}, ncost: {}"
                  .format(init_no + 1, itr, max_iter, moves, ncost))

    return centroids, labels, cost, itr, epoch_costs, newly_initialized_cluster


def _k_prototypes_iter(Xnum, Xcat, Xlist, Xtime, centroids, cl_attr_sum_num, cl_memb_sum, cl_attr_freq_cat, cl_attr_freq_list ,  cl_attr_freq_time,
                       membship, num_dissim, cat_dissim, list_dissim, time_dissim, time_max_values, gamma, random_state, verbose=True, n_points=0):
    """ Single iteration of the adjusted k-Prototypes

    :param Xnum, Xcat, Xlist, Xtime:                                                Different parts of the dataset
    :param centroids:                                                               Current centroids for k-Prototypes
    :param cl_memb_sum:                                                             How many datapoints each clusters has 
    :param cl_attr_sum_num, cl_attr_freq_cat, cl_attr_freq_list cl_attr_freq_time   Sums and Frequencies of each Feature in each cluster
    :param membship:                                                                Current point-to-cluster association
    :param num_dissim, cat_dissim, list_dissim, time_dissim:                        Different distance functions
    :param gamma:                                                                   Weighting factor for non-numerical values
    :param random_state:                                                            Random state for random number generation
    :param verbose:                                                                 Verbosity mode.
    :param n_points:                                                                Number of datapoints

    :returns updated centroids and labels
    
    """
    
    moves = 0
    if verbose:
        print("Running Iteration:")

    for ipoint in tqdm(range(n_points), disable= not verbose):
        num_distances=num_dissim(centroids[0], Xnum[ipoint]) if Xnum.size!=0 else 0
        cat_distances=cat_dissim(centroids[1], Xcat[ipoint], X=Xcat, membship=membship) if Xcat.size!=0 else 0
        list_distances=list_dissim(centroids[2], Xlist[ipoint]) if Xlist.size!=0 else 0
        time_distances=time_dissim(centroids[3], Xtime[ipoint], time_max_values=time_max_values)if Xtime.size!=0 else 0
        clust = np.argmin(num_distances + gamma * (cat_distances + list_distances + time_distances))
            
        #clust = np.argmin(costs_array)
        if membship[clust, ipoint]:
            # Point is already in its right place.
            continue

        # Move point, and update old/new cluster frequencies and centroids.
        moves += 1
        old_clust = np.argwhere(membship[:, ipoint])[0][0]

        # Moved a point, update association!
        cl_memb_sum[clust] += 1
        cl_memb_sum[old_clust] -= 1
        membship[clust, ipoint] = 1
        membship[old_clust, ipoint] = 0

        if Xnum.size!=0:
            _move_point_num(Xnum[ipoint], clust, old_clust, cl_attr_sum_num, cl_memb_sum, centroids[0])
        if Xcat.size!=0:
            _move_point_cat(Xcat[ipoint], clust, old_clust, cl_attr_freq_cat, centroids[1])
        if Xlist.size!=0:
            _move_point_list(Xlist[ipoint], clust, old_clust, cl_attr_freq_list, centroids[2])
        if Xtime.size!=0:
            _move_point_cat(Xtime[ipoint], clust, old_clust, cl_attr_freq_time, centroids[3])

        # In case of an empty cluster, reinitialize with a random point from largest cluster.
        if not cl_memb_sum[old_clust]:
            from_clust = membship.sum(axis=1).argmax()
            choices = [ii for ii, ch in enumerate(membship[from_clust, :]) if ch]
            rindx = random_state.choice(choices)
            
            # Moved a point, update association!
            cl_memb_sum[old_clust] += 1
            cl_memb_sum[from_clust] -= 1
            membship[old_clust, rindx] = 1
            membship[from_clust, rindx] = 0

            if Xnum.size!=0:
                _move_point_num(Xnum[rindx], old_clust, from_clust, cl_attr_sum_num, cl_memb_sum, centroids[0])

            if Xcat.size!=0:
                _move_point_cat(Xcat[rindx], old_clust, from_clust, cl_attr_freq_cat, centroids[1])
            
            if Xlist.size!=0:
                _move_point_list(Xlist[rindx], old_clust, from_clust, cl_attr_freq_list, centroids[2])

            if Xtime.size!=0:
                _move_point_cat(Xtime[rindx], old_clust, from_clust, cl_attr_freq_time, centroids[3])

    return centroids, moves, membship


def adjusted_ch_index(Xnum, Xcat, Xlist, Xtime, centroids, num_dissim, cat_dissim, list_dissim, time_dissim, time_max_values, gamma, labels, intra_cs, n_clusters):
    """ Calculade the CH Index, adjusted for Categorical, List and Time types.
    Implementation based on ref: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html

    :param Xnum, Xcat, Xlist, Xtime:                            Different parts of the dataset
    :param centroids:                                           Current centroids for k-Prototypes
    :param num_dissim, cat_dissim, list_dissim, time_dissim:    Different distance functions
    :param gamma:                                               Weighting factor for non-numerical values
    :param labels:                                              Current cluster labels
    :param intra_cs:                                            Intra Cluster distances (basically total clustering costs)
    :param n_clusters:                                          Number of clusters to calculate

    :returns the adjusted CH Index value  
    
    """
    n_samples = labels.shape[0]
    extra_cs = 0.0
    
    total_mean_num = np.mean(Xnum, axis=0) if Xnum.size!=0 else np.empty(0)
    total_mean_categorical = find_max_frequency_attribute(Xcat)
    total_centroids_lists = find_max_frequency_attribute(Xlist, True)
    total_centroids_time = find_max_frequency_attribute(Xtime)

    _, counts = np.unique(labels, return_counts=True)
    for i in range(n_clusters):   
        num_costs = num_dissim(np.expand_dims(centroids[0][i], axis=0), total_mean_num) if total_mean_num.size > 0 else 0
        cat_costs = cat_dissim(np.expand_dims(centroids[1][i], axis=0), total_mean_categorical) if total_mean_categorical.size > 0 else 0
        lists_costs = list_dissim(np.expand_dims(centroids[2][i], axis=0), total_centroids_lists) if total_centroids_lists.size > 0 else 0
        time_costs = time_dissim(np.expand_dims(centroids[3][i], axis=0), total_centroids_time, time_max_values=time_max_values) if total_centroids_time.size > 0 else 0
        tot_costs = num_costs + gamma[i] * (cat_costs + lists_costs + time_costs)
        extra_cs += counts[i] * tot_costs[0]

    return (
        1.0
        if intra_cs == 0.0
        else extra_cs * (n_samples - n_clusters) / (intra_cs * (n_clusters - 1.0))
    )

def costs(Xnum, Xcat, Xlists, Xtime, centroids, num_dissim, cat_dissim, lists_dissim, time_dissim, time_max_values, gamma, membship=np.empty(0), labels=np.empty(0), n_clusters=-1):
    """ Calculate the costs for the given centroids and datapoints

    :param Xnum, Xcat, Xlist, Xtime:                            Different parts of the dataset
    :param centroids:                                           Current centroids for k-Prototypes
    :param num_dissim, cat_dissim, list_dissim, time_dissim:    Different distance functions
    :param gamma:                                               Weighting factor for non-numerical values
    :param membship:                                            Current point-to-cluster association
    :param labels:                                              Current point-to-cluster association (different format for reusability)
    :param n_clusters:                                          Number of clusters to calculate

    :returns total cost for the given clusters
    
    """

    total_costs = 0.
    # For each cluster
    for i in range(n_clusters):
        # 1. Get all points associated to each cluster
        if membship.size!=0:
            point_indices = np.argwhere(membship[i]).flatten()
        elif labels.size!=0:
            point_indices = np.argwhere(i == labels).flatten()
        else:
            raise RuntimeError("Both membship and lables argument were empty")
        num_costs = np.sum(num_dissim(Xnum[point_indices], centroids[0][i])) if Xnum.size!=0 else 0
        cat_costs = np.sum(cat_dissim(Xcat[point_indices], centroids[1][i], X=Xcat[point_indices], membship=membship)) if Xcat.size!=0 else 0
        lists_costs = np.sum(lists_dissim(Xlists[point_indices], centroids[2][i])) if Xlists.size!=0 else 0
        time_costs = np.sum(time_dissim(Xtime[point_indices], centroids[3][i], time_max_values=time_max_values)) if Xtime.size!=0 else 0

        total_costs += num_costs + gamma[i] * (cat_costs + lists_costs + time_costs)

    return total_costs

def labels_cost(Xnum, Xcat, Xlists, Xtime, centroids, num_dissim, cat_dissim, lists_dissim, time_dissim, time_max_values, gamma, n_points, verbose=True, membship=None):
    """ Calculate the new labels and the cost of the current centroids and datapoints.

    :param Xnum, Xcat, Xlist, Xtime:                            Different parts of the dataset
    :param centroids:                                           Current centroids for k-Prototypes
    :param num_dissim, cat_dissim, list_dissim, time_dissim:    Different distance functions
    :param gamma:                                               Weighting factor for non-numerical values
    :param n_points:                                            Number of datapoints
    :param verbose:                                             Verbosity mode.
    :param membship:                                            Current point-to-cluster association
    
    :returns total cost for the given clusters
    
    """
    cost = 0.
    labels = np.empty(n_points, dtype=np.uint16)
    if verbose:
        logging.info(f"\nLabels Cost Calculation")
    for ipoint in tqdm(range(n_points), mininterval=.5, disable= not verbose):
        num_costs = num_dissim(centroids[0], Xnum[ipoint]) if Xnum.size!=0 else 0
        cat_costs = cat_dissim(centroids[1], Xcat[ipoint], X=Xcat, membship=membship) if Xcat.size!=0 else 0
        lists_costs = lists_dissim(centroids[2], Xlists[ipoint]) if Xlists.size!=0 else 0
        time_costs = time_dissim(centroids[3], Xtime[ipoint], time_max_values=time_max_values) if Xtime.size!=0 else 0

        # Gamma relates the categorical cost to the numerical cost.
        tot_costs = num_costs + gamma * (cat_costs + lists_costs + time_costs)
        # It seems a little inefficient to calculate everything, instead of just the costs for the associated clusters....
        clust = np.argmin(tot_costs)

        labels[ipoint] = clust
        cost += tot_costs[clust]
    return labels, cost

def _move_point_num(point, to_clust, from_clust, cl_attr_sum_num, cl_memb_sum, centroids):
    """ Move point between clusters, numerical attributes
    
    :param point:           Data point
    :param to_clust:        New cluster
    :param from_clust:      Old cluster
    :param cl_attr_sum_num: Sum of attribute values within the cluster
    :param cl_memb_sum:     How many datapoints each clusters has 
    :param centroids:       Current centroids for k-Prototypes
    :returns updated centroids and new attribute sum
    """

    for iattr, curattr in enumerate(point):
        cl_attr_sum_num[to_clust, iattr] += curattr
        cl_attr_sum_num[from_clust, iattr] -= curattr

    for iattr in range(len(point)):
        for curc in (to_clust, from_clust):
            if cl_memb_sum[curc]:
                centroids[curc, iattr] = cl_attr_sum_num[curc, iattr] / cl_memb_sum[curc]
            else:
                centroids[curc, iattr] = 0.

def _move_point_cat(point, to_clust, from_clust, cl_attr_freq, centroids):
    """ Move point between clusters, categorical attributes
    
    :param point:           Data point
    :param to_clust:        New cluster
    :param from_clust:      Old cluster
    :param cl_attr_freq:    Frequencies of attribute values within the cluster
    :param centroids:       Current centroids for k-Prototypes
    :returns updated centroids and new attribute frequencies
    """

    for iattr, curattr in enumerate(point):
        to_attr_counts = cl_attr_freq[to_clust][iattr]
        from_attr_counts = cl_attr_freq[from_clust][iattr]

        # Increment the attribute count for the new "to" cluster
        to_attr_counts[curattr] += 1
        # Decrement the attribute count for the old "from" cluster
        from_attr_counts[curattr] -= 1

        current_attribute_value_freq = to_attr_counts[curattr]
        current_centroid_value = centroids[to_clust][iattr]
        current_centroid_freq = to_attr_counts[current_centroid_value]
        if current_centroid_freq < current_attribute_value_freq:
            # We have incremented this value to the new mode. Update the centroid.
            centroids[to_clust][iattr] = curattr

        old_centroid_value = centroids[from_clust][iattr]
        if old_centroid_value == curattr:
            # We have just removed a count from the old centroid value. We need to
            # recalculate the centroid as it may no longer be the maximum
            centroids[from_clust][iattr] = get_max_value_key(from_attr_counts)

def _move_point_list(point, to_clust, from_clust, cl_attr_freq, centroids):
    """ Move point between clusters, list attributes
    
    :param point:           Data point
    :param to_clust:        New cluster
    :param from_clust:      Old cluster
    :param cl_attr_freq:    Frequencies of attribute values within the cluster
    :param centroids:       Current centroids for k-Prototypes
    :returns updated centroids and new attribute frequencies
    """
    for iattr, curattrs in enumerate(point):
        to_attr_counts = cl_attr_freq[to_clust][iattr]
        from_attr_counts = cl_attr_freq[from_clust][iattr]
        
        str_attr = str(curattrs)
        to_attr_counts[str_attr] += 1
        # Decrement the attribute count for the old "from" cluster
        from_attr_counts[str_attr] -= 1

        current_attribute_value_freq = to_attr_counts[str_attr]
        current_centroid_value = centroids[to_clust][iattr]
        current_centroid_freq = to_attr_counts[str(current_centroid_value)]
        # Increment the attribute count for the new "to" cluster
        if current_centroid_freq < current_attribute_value_freq:
            # We have incremented this value to the new mode. Update the centroid.
            centroids[to_clust][iattr] = curattrs


        old_centroid_value = centroids[from_clust][iattr]
        if old_centroid_value == str_attr:
            # We have just removed a count from the old centroid value. We need to
            # recalculate the centroid as it may no longer be the maximum
            centroids[from_clust][iattr] = eval(get_max_value_key(from_attr_counts), )

def _update_gamma_values(gamma, Xnum, membship, n_clusters):
    """ Updates gamma values for each cluster based on the standard deviation within the data points in each cluster.

    :param gamma:       Current gamma values per cluster
    :param Xnum:        Data set numerical columns
    :param membship:    Current point-to-cluster association

    :returns updated gamma values
    """

    if Xnum.size == 0:
        return gamma

    gamma = deepcopy(gamma)
    for i in range(n_clusters):
        # 1. Get all points associated to each cluster
        point_indices = np.argwhere(membship[i]).flatten()

        # 2. Get std of all Xnum
        associated_points = Xnum[point_indices]
        if associated_points.shape[0] == 0:
            continue
        else:
            gamma[i] = associated_points.std() * 0.5

    return gamma

    
 