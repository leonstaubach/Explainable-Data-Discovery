import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute()))

from src.utils import get_type_through_index
from config import FEATURE_IMPORTANCE_METRIC, ANONIMIZE_FEATURE_NAMES
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_array

"""
    Formula to calculate feature importance contains two main metrics:

    1. Intra-Cluster-Feature Distance (Higher is worse)

    Idea: See how close all associated points are to their respective cluster centroid
    Pseudo Code:
    - for each cluster
        - set data_i = all data points associated to that cluster
        - set current_cluster_i = clusters[i]
        - for each feature
            - calculate avg similarity between data_i[feature] and current_cluster_i[feature] 

    2. Extra-Cluster-Feature Contribution (Higher is better)

    Idea: See how far away the other points (that are not associated to the cluster) of the data set are from the cluster centroid
    Pseudo Code:
    - for each cluster
        - set data_non_i = all data points that are NOT associated to that cluster
        - set current_cluster_i = clusters[i]
        - for each feature
            - calculate avg (dis)similarity between data_non_i[feature] and current_cluster_i[feature]


    Problem: what if the cluster makes up 90% -> 10% of data remaining -> distance will be smalller
    --> In general that is not a big problem, because i only compare features within a cluster (therefore all features within the cluster are affected the same way)



    Formula Discussion:
    1. return avg_extra_cs (higher -> better, optimal: 1 (farthest away), worst: 0 (closest to all points))
        -> doesn't directly reflect how the distribution of the feature in the cluster is
    2. return avg_intra_cs (lower -> better, optimal: 0 (closest), worst: 1 (farthest away))
        -> doesn't directly reflect how the cluster is integrated into the larger feature-context
    3. return formula+1 = avg_extra_cs / (avg_intra_cs+1) (higher -> better, optimal: 1, worst: 0)
        -> Not biased!
    4. return formula_normalized = avg_extra_cs / (avg_intra_cs + avg_extra_cs)
        -> Unidirected dependency issues for intra cs on extra cs (e.g. if avg_extra_cs=0, the value of avg_intra_cs doesn't matter)
    
    """


def create_explicit_feature_importance(indices_map, labels, columns, centroids, df, n_clusters, distance_functions, return_print:bool = True):
    """ Calculate feature importance scores for each feature in each cluster.

    :param indices_map:         Indices for each used feature
    :param labels:              Labels for each data point of the cluster run
    :param columns:             Columns that describe the data
    :param centroids:           Centroids of the cluster run       
    :param df:                  Dataframe with metainformation about differnt columns
    :param n_clusters:          Number of clusters
    :param distance_functions:  List of usable distance function to calculate similarities
    :param return_print:        Whether to return a printable table or a usable pandas table

    :returns either a prettified list of strings or the actual pandas table.
    
    """
    X = df.values
    columns = np.array(columns)
    feature_names = []
    flattened_feature_indices = []
    for name, indices in indices_map.items():
        if name == "time_max_values":
            continue
        feature_names.append(columns[indices])
        flattened_feature_indices.append(indices)

    feature_names = [item for sublist in feature_names for item in sublist]
    flattened_feature_indices = [item for sublist in flattened_feature_indices for item in sublist]

    
    intermediate_result = []
    # For each cluster
    for i in range(n_clusters):
        cluster_result = {}
        

        associated_points = np.argwhere(i == labels).flatten()
        mask = np.ones(X.shape[0], bool)
        
        mask[associated_points] = False

  
        # All associated points
        data_i = X[associated_points]

        # All other points
        non_data_i = X[mask]

        # For each feature
        seen = np.zeros((4,), dtype=np.int16)
        for j in flattened_feature_indices:
            extra_cs = 0.0
            indx = get_type_through_index(indices_map, j)
            feature_name = columns[j]

            # Function to be applied to the data to calculate similarity
            distance_function = distance_functions[indx]
            # Centroid value of the current feature
            feature_centroid = np.expand_dims(centroids[indx][i][seen[indx]], axis=0)

            # Feature values of the current cluster
            current_feature_values = np.expand_dims(data_i[:, j], axis=0)
        
            # Feature values of points that are not associated to the current cluster
            other_feature_values = np.expand_dims(non_data_i[:, j], axis=0)

            # Check if numerical
            if isinstance(data_i[0, j], float):
                current_feature_values = check_array(current_feature_values) 
                other_feature_values = check_array(other_feature_values) 

            cyclic_lookup = [indices_map["time_max_values"][seen[indx]]] if indx == 3 else None
            intra_cs = (distance_function(current_feature_values, feature_centroid, time_max_values=cyclic_lookup) / current_feature_values.shape[1])[0]
            extra_cs = (distance_function(other_feature_values, feature_centroid, time_max_values=cyclic_lookup) / other_feature_values.shape[1])[0]

            seen[indx] += 1
            cluster_result[feature_name] = {
                "intra_cs": intra_cs,
                "extra_cs": extra_cs,
                "formula+1": extra_cs/(intra_cs+1),
                "formula_normalized": extra_cs/(extra_cs+intra_cs),
                "formula_updated": max(extra_cs-intra_cs, 0)/ extra_cs if extra_cs > 0 else 0
            }

        intermediate_result.append(cluster_result)
    if not return_print:
        return intermediate_result
    result = []
    for i, map in enumerate(intermediate_result):
        map = {feature_name: values for feature_name, values in sorted(map.items(), key=lambda item: item[1][FEATURE_IMPORTANCE_METRIC], reverse=True)}
        pdMap = pd.DataFrame(map)
        columns = list(columns)
        if ANONIMIZE_FEATURE_NAMES:
            pdMap.columns = [f"Feature {columns.index(feature_name)}" for feature_name in pdMap.columns]

        result.append(pdMap.to_markdown())
    return result