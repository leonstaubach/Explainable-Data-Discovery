from copy import deepcopy
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute()))

import logging
import numpy as np
import pandas as pd
from collections import defaultdict


import config
from src.data_loader import CustomDataset
from src.visualisation_utils import create_linecharts_elbow_method
import src.utils as utils
from src.kmodes.util import get_max_value_key
from src.kmodes.kprototypes import KPrototypes, adjusted_ch_index, costs
from multiprocessing import cpu_count

"""
This part of the process analyzes the feature importance based on the distribution of the data and the feature ranking from the previous step of the process.
Some responsibilities are:
- propose a proker k (Number of Clusters)
 -- For this: run k-prototypes for k in {2, ..., 10}
 -- Plot k over costs and a cluster-quality measurement (CH-Index, alternatively Silhouette Score)
 -- Choose k the first peak in cluster-quality
 -- ref: https://medium.com/@haataa/how-to-measure-clustering-performances-when-there-are-no-ground-truth-db027e9a871c

- run k-Prototypes on whole dataset
- check each equal-valued feature (each feature that yields the same values for all centroids)
- start removing features based on the given ranking
- test to add all removed features (except the latest)
- return the optimal feature-set

The main comparison technique is the calculation of the normalized CH-Index (DOI: 10.1007/springerreference_302701).
"""


def elbow_method(df: pd.DataFrame, max_k: int, indices_map: dict):
    """ Runs Elbow method. Uses CH index to heuristically find a good number of clusters.

    :param df:                  Given dataframe with metainformation about differnt columns
    :param max_k:               Maximum number of clusters
    :param indices_map:         Index of columns that contain numerical, categorical, list and time data

    :returns   the optimal k measured by highest CH index
    """
    range_k = list(range(2, max_k+1))
    costs = []
    ch_indeces = []
    n_jobs = min(cpu_count(), config.K_PROTOTYPE_REPEAT_NUM)
    verbose = False if n_jobs > 1 else True
    for k in range_k:
        k_prototype = KPrototypes(n_clusters=k, init='Cao', n_init=config.K_PROTOTYPE_REPEAT_NUM, verbose=verbose, n_jobs=n_jobs)
        _, cost , ch_index = k_prototype.fit_predict(df, indices_map=indices_map)
        costs.append(cost)

        ch_indeces.append(ch_index)
        logging.info(f"\nFinished Elbow Test for k={k}, resulted in costs: {cost}\tch-index: {ch_index}")
    # Alternatively:
    # chosen_k = range_k[utils.first_peak(ch_indeces)]
    chosen_k = range_k[ch_indeces.index(max(ch_indeces))]
    if config.STORE_IMAGES_DURING_EXECUTION or config.SHOW_IMAGES_DURING_EXECUTION:
        utils.check_dir_file(config.PATH_OUTPUT_IMAGES, False)
        path = f"{config.PATH_OUTPUT_IMAGES}/2_0_Elbow_Method.png"
        create_linecharts_elbow_method(range_k, costs, ch_indeces, config.STORE_IMAGES_DURING_EXECUTION, config.SHOW_IMAGES_DURING_EXECUTION, path)
    
    return chosen_k

def calculate_normalized_ch_index(X, ch_index_i, feature_set_j, k_proto_obj_i: KPrototypes, use_old_gamma: bool=False):
    """ Calculates the normalized CH index for a given feature_set_j and given labels_i (from two different clustering runs)
    ref DOI: 10.1007/springerreference_302701

    :param X                The full dataset
    :param ch_index         The ch_index of run i
    :param feature_set_j    The set of features for the different run j
    :param k_proto_obj_i    The k-Prototypes object containing the labels from run i
    :param use_old_gamma    Whether to recalculate gamma based on the new dataset (comparability vs optimization)

    :returns    normalized CH-Index for given run i 
    """
    # Use labels from i to create new centroids for feature_set_j
    n_clusters = k_proto_obj_i.n_clusters
    
    # Same for the membership sum per cluster
    cl_memb_sum = np.zeros(n_clusters, dtype=int)

    cl_attr_sum_num = np.zeros((n_clusters, len(feature_set_j["num_indices"])), dtype=np.float64)

    # cl_attr_freq is a list of lists with dictionaries that contain
    # the frequencies of values per cluster and attribute.
    cl_attr_freq_cat = [[defaultdict(int) for _ in range(len(feature_set_j["cat_indices"]))]
                    for _ in range(n_clusters)]

    cl_attr_freq_list = [[defaultdict(int) for _ in range(len(feature_set_j["list_indices"]))]
                    for _ in range(n_clusters)]

    cl_attr_freq_time = [[defaultdict(int) for _ in range(len(feature_set_j["time_indices"]))]
                    for _ in range(n_clusters)]

    # Use the feature set of j, otherwise all information from i
    Xnum, Xcat, Xlist, Xtime = utils.split_num_cat(X, feature_set_j)
    for clust in range(k_proto_obj_i.n_clusters):

        associated_points = np.argwhere(clust == k_proto_obj_i.labels_).flatten()
        cl_memb_sum[clust] = associated_points.shape[0]
        if Xnum.size > 0: 
            for data_point in Xnum[associated_points]:
                for iattr, curattr in enumerate(data_point):
                    cl_attr_sum_num[clust, iattr] += curattr
        
        if Xcat.size > 0: 
            for data_point in Xcat[associated_points]:
                for iattr, curattr in enumerate(data_point):
                    cl_attr_freq_cat[clust][iattr][curattr] += 1
        
        if Xlist.size > 0: 
            for data_point in Xlist[associated_points]:
                for iattr, curattr_list in enumerate(data_point):
                    # Transform cur_attr_list to str, to count frequency of occurence
                    cl_attr_freq_list[clust][iattr][str(curattr_list)] +=1
        
        if Xtime.size > 0:
            for data_point in Xtime[associated_points]:
                for iattr, curattr in enumerate(data_point):
                    # Transform cur_attr_list to str, to count frequency of occurence
                    cl_attr_freq_time[clust][iattr][curattr] +=1
    
    centroids = [np.empty((n_clusters, len(feature_set_j["num_indices"])), dtype=np.float64),
                    np.empty((n_clusters, len(feature_set_j["cat_indices"])), dtype=object),
                    np.empty((n_clusters, len(feature_set_j["list_indices"])), dtype=object),
                    np.empty((n_clusters, len(feature_set_j["time_indices"])), dtype=object)]

    # Perform an centroid update calculation.
    for ik in range(n_clusters):
        for iattr in range(len(feature_set_j["num_indices"])):
            # I think this basically averages the centroid value to the avg member attribute value
            centroids[0][ik, iattr] = cl_attr_sum_num[ik, iattr] / cl_memb_sum[ik]
        for iattr in range(len(feature_set_j["cat_indices"])):
            # Argument are the frequencies of the unique values for this attribute in the cluster
            # Returns the maximum category in the given cluster (most frequent category builds the new cluster centroid)
            centroids[1][ik, iattr] = get_max_value_key(cl_attr_freq_cat[ik][iattr])
            
        for iattr in range(len(feature_set_j["list_indices"])):
            centroids[2][ik, iattr] = eval(get_max_value_key(cl_attr_freq_list[ik][iattr]))

        for iattr in range(len(feature_set_j["time_indices"])):
            centroids[3][ik, iattr] = get_max_value_key(cl_attr_freq_time[ik][iattr])

    # Recalculate gamma on changed feature set
    if use_old_gamma:
        gamma = k_proto_obj_i.gamma
    else:
        gamma = utils.initialize_gamma(Xnum, n_clusters)
    logging.info(f"Using new gamma here: {gamma}")
    cost = costs(Xnum, Xcat, Xlist, Xtime, centroids, k_proto_obj_i.num_dissim, \
        k_proto_obj_i.cat_dissim, k_proto_obj_i.list_dissim, k_proto_obj_i.time_dissim, feature_set_j["time_max_values"], gamma, labels=k_proto_obj_i.labels_, n_clusters=n_clusters)

    ch_feature_set_j = adjusted_ch_index(Xnum, Xcat, Xlist, Xtime, centroids, k_proto_obj_i.num_dissim, \
        k_proto_obj_i.cat_dissim, k_proto_obj_i.list_dissim, k_proto_obj_i.time_dissim, feature_set_j["time_max_values"], gamma, \
        k_proto_obj_i.labels_, cost, n_clusters)
    logging.info(f"\nCH_INDEX_I: {ch_index_i} multiplied with {ch_feature_set_j}")
    return ch_index_i * ch_feature_set_j


def execute_iterative_clustering(df: pd.DataFrame = pd.DataFrame(), meta_data: dict = {}, use_old: bool = False):
    """ Executes step 2 of the process described in the main paper. Contains steps like
            - initial elbow method to determine the number of clusters k
            - initial clustering
            - checking for elimination of each feature with equal centroid values
            - backward elimination from a given subset of features
            - reconsidering eliminated features to find the optimal feature subset

    :param df:          Given dataframe with metainformation about differnt columns
    :param meta_dict:   Meta Data for the given run (like prestored k)
    :param use_old:     Whether to use the old feature rankig or the new (old: proposed in paper, new: my own implementation)

    :returns   optimal feature subset
    """

    indices_map = utils.create_indices_map(df.columns, df.attrs)

    # Determine k through Elbow method if it is not given.
    if "k" not in meta_data:
        k = elbow_method(df, config.MAX_NUM_CLUSTERS_ELBOW, indices_map)
        utils.check_dir_file(config.PATH_OUTPUT_METADATA, True)
        meta_data["k"] = k
        utils.update_meta_data_attribute(f"{config.PATH_OUTPUT_METADATA}", "k", k)
    else:
        k = meta_data.get("k")

    key_to_use = "feature_ranking" if not use_old else "old_feature_ranking"

    if key_to_use not in meta_data:
        raise RuntimeError("Didn't provide a 'feature_ranking' attribute in the meta data")

    feature_ranking = meta_data[key_to_use]
    columns = list(df.columns)

    # Start calculating 
    n_jobs = min(cpu_count(), config.K_PROTOTYPE_REPEAT_NUM)
    verbose = False if n_jobs > 1 else True
    logging.info("\nInitial Cluster Calculation for Feature Selection")
    k_prototype = KPrototypes(n_clusters=k, init='Cao', n_init=config.K_PROTOTYPE_REPEAT_NUM, verbose=verbose, n_jobs=n_jobs)
    _, _, ch_index = k_prototype.fit_predict(df, indices_map=indices_map)

    X = df.values

    all_equal_features = []
    """ Find each feature that has all equal values in centroids """
    for type_index, data_type_clusters in enumerate(k_prototype.cluster_centroids_):
        if data_type_clusters.shape[1] == 0:
            continue
        
        for internal_feature_index in range(len(data_type_clusters[0])):
            prev = None
            prev_different = False
            for cluster_index in range(len(data_type_clusters)):
                if prev == None:
                    prev = data_type_clusters[cluster_index][internal_feature_index]

                elif data_type_clusters[cluster_index][internal_feature_index] != prev:
                    prev_different = True
                    break

            if not prev_different:
                # Resolve internal feature index to actual feature index
                actual_feature_index = indices_map[utils.resolve_index(type_index)][internal_feature_index]
                all_equal_features.append(actual_feature_index)
             
    checkable_column_names = [columns[v] for v in all_equal_features]  
    # Sort this by feature ranking
    checkable_column_names = [v for v in feature_ranking if v in checkable_column_names]    
    
    original_indices = deepcopy(indices_map)
    best_ch_index = ch_index
    best_k_prototype = k_prototype
    removed_features = []

    if len(checkable_column_names) == 0:
        logging.info(f"\nDidn't find any feature with all-equal centroid values across all clusters.")
        
    else:
        logging.info(f"\nFound the following features with equal centroid values: {checkable_column_names}")
        while len(checkable_column_names) > 0:
            
            # Pop the lowest ranked feature from the set
            indices_map_next, removed_feature, new_centroids = utils.remove_feature_from_indices_map(checkable_column_names, columns, indices_map, best_k_prototype.newly_initialized_centroids)
            logging.info(f"\n Running clustering with popped feature {removed_feature}")
            
            # Rerun k_prototypes with lower set, n_init=1, because it is initialized with existing centroids for comparability
            k_prototype_next = KPrototypes(n_clusters=k, n_init=1, verbose=True, n_jobs=n_jobs, initial_centroids=new_centroids)
            _, _, ch_index_next = k_prototype_next.fit_predict(df, indices_map=indices_map_next)
        
            # Calculate normalized CH Index 
            normalized_ch_index = calculate_normalized_ch_index(X, best_ch_index, indices_map_next, best_k_prototype, False)
            normalized_ch_index_next = calculate_normalized_ch_index(X, ch_index_next, indices_map, k_prototype_next, False)
        
            logging.info(f"\nFullSet: {indices_map} -> NCH={normalized_ch_index} against\nRemovedSet: {indices_map_next} -> NCH={normalized_ch_index_next}")

            # If the reduced solution is atleast 99.99% as good (to handle minor movements through initialization or equal values): delete the feature 
            if normalized_ch_index_next >= 0.9999*normalized_ch_index:
                logging.info(f"\nRemoved Feature {removed_feature}")  

                indices_map = indices_map_next
                best_ch_index = deepcopy(ch_index_next)
                best_k_prototype = deepcopy(k_prototype_next) 
                removed_features.append(feature_ranking.pop(feature_ranking.index(removed_feature)))
            else:
                logging.info(f"\nDid not remove the feature")     
    
        # If the last ranked feature did not get removed at this step, we don't need to check it on the next iteration.
        if removed_feature == feature_ranking[-1]:
            feature_ranking.pop()
    
    # Just reset to not confuse, since all features were checked
    all_equal_features.clear()
    
    logging.info("\nStarting to remove features iteratively")
    
    # Iterative Step, remove features from the given feature ranking, until it yields no improvement.
    while len(columns) - len(removed_features) > 2:
        # Pop the lowest ranked feature from the set
        indices_map_next, removed_feature, new_centroids = utils.remove_feature_from_indices_map(feature_ranking, columns, indices_map, best_k_prototype.newly_initialized_centroids)
        logging.info(f"\n Running clustering with popped feature {removed_feature}")
        
        # Rerun k_prototypes with lower set, n_init=1, because it is initialized with existing centroids for comparability
        k_prototype_next = KPrototypes(n_clusters=k, init='Cao', n_init=1, verbose=True, n_jobs=n_jobs, initial_centroids=new_centroids)
        _, _, ch_index_next = k_prototype_next.fit_predict(df, indices_map=indices_map_next)
    
        # Calculate normalized CH Index 
        normalized_ch_index = calculate_normalized_ch_index(X, best_ch_index, indices_map_next, best_k_prototype, False)
        normalized_ch_index_next = calculate_normalized_ch_index(X, ch_index_next, indices_map, k_prototype_next, False)
        
        logging.info(f"\nFullSet: {indices_map} -> NCH={normalized_ch_index} against\nRemovedSet: {indices_map_next} -> NCH={normalized_ch_index_next}")

        # If the reduced solution is atleast 99.99% as good (to handle minor movements through initialization or equal values): delete the feature 
        if normalized_ch_index_next >= 0.9999*normalized_ch_index:
            removed_features.append(removed_feature)
            indices_map = indices_map_next
            
            best_ch_index = deepcopy(ch_index_next)
            best_k_prototype = deepcopy(k_prototype_next)
            logging.info("\nAnother iteration, because NCHnext >= NCHcurr")

        # Assume that we don't need to check further, since the 'worst' feature (with regards to redundancy) did not improve clustering quality
        else:
            break

    # Ignore the latest removed index for now
    if len(removed_features) == 0:
        logging.info(f"\nAfter eliminating feature-by-feature, there was no insignificant feature removed, so the current solution is likely to be optimal")
        removed_features.append(None)

    removed_features = removed_features[:-1]
    """ Consider finding the optimal set from the remaining and removed features
     --> Greedy approach from earlier may loose subset tests that haven't been tested before
     --> Add removed features one by one to the remaining feature set
     ---> Compare to best seen feature set (by NCH)
     ---> If better: update feature set, best NCH and removed feature set -> repeat until removed features have been seen
    """
    logging.info("\nStarting to add back features iteratively")
    
    while True:
        if len(removed_features) == 0:
            logging.info("\nNo more indices to check, breaking out of the loop...")
            break
        logging.info(f"\nRemaining features to check: {removed_features}")
        added_feature = removed_features.pop(0)
        logging.info(f"\nRunning clustering with added feature {added_feature}")
        
        indices_map_next, new_centroids = utils.add_feature_to_indices_map(original_indices, added_feature, columns, indices_map, best_k_prototype.newly_initialized_centroids, k_prototype.newly_initialized_centroids)

        # Rerun k_prototypes with lower set, n_init=1, because it is initialized with existing centroids for comparability
        try:
            k_prototype_next = KPrototypes(n_clusters=k, init='Cao', n_init=1, verbose=True, n_jobs=n_jobs, initial_centroids=new_centroids)
            _, _, ch_index_next = k_prototype_next.fit_predict(df, indices_map=indices_map_next)

        # Happens when clusters cannot be properly initialized (when the number of unique combinations is lower than the cluster num)
        except utils.WouldTryRandomInitialization:
            continue
        # Calculate CH Index
        normalized_ch_index = calculate_normalized_ch_index(X, best_ch_index, indices_map_next, best_k_prototype)
        normalized_ch_index_next = calculate_normalized_ch_index(X, ch_index_next, indices_map, k_prototype_next)
        logging.info(f"\nOptimalSet: {indices_map} -> NCH={normalized_ch_index} against\nAddedSet: {indices_map_next} -> NCH={normalized_ch_index_next}")

        # If the reduced solution is atleast 0.01% better: add the feature back
        if normalized_ch_index_next >= 1.0001*normalized_ch_index:
            indices_map = indices_map_next
            best_ch_index = deepcopy(ch_index_next)
            best_k_prototype = deepcopy(k_prototype_next)
            logging.info(f"\nAdded feature {added_feature}")
        else:
            logging.info(f"\nDid not add feature {added_feature}")
    
    logging.info(f"\nThe final optimal feature subset is {indices_map}")

    return indices_map


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    utils.check_dir_file(config.PATH_OUTPUT_METADATA, True)
    meta_data = utils.load_meta_data_attribute(config.PATH_OUTPUT_METADATA)
    dataset = CustomDataset(config.create_processed_data_path(True))

    execute_iterative_clustering(dataset.get_training_df(), meta_data)