import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute()))

import pandas as pd
import numpy as np
import os
import pyarrow.parquet as pq
from tqdm import tqdm
import numbers
import time
import logging
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from copy import deepcopy

from src.data_loader import CustomDataset
import src.utils as utils
import src.visualisation_utils as vis_utils
import config

""" Overview
This part of the process analyzes the general distribution of the data. Some responsibilities are
- eliminating noisy features (e.g. features with only one unique value)
- visualizing general data distribution and joint probabilities
- calculating normalized mutual information between each unique feature-pair
- ranking features based on remaining highest mutual information (redundancy ranking)

"""

def prepare_data_for_mutual_information(df: pd.DataFrame) -> np.ndarray:
    """ Transform df data to numpy. Also apply binning to numerical values.

    :param df:      The given dataframe

    :returns numpy representation of data
    """
    numpy_data = df.to_numpy()

    for i, col in enumerate(df.columns):
        if df.attrs[col]["datatype"] == numbers.Number:
            returned_data, interval_width, number_of_bins = utils.equal_width_binning(numpy_data[:, i])
            numpy_data[:, i] = returned_data
            df.attrs[col]["interval_width"] = interval_width
            df.attrs[col]["number_of_bins"] = number_of_bins
    return numpy_data


def feature_entropy(data: np.ndarray, df_attrs: dict, columns: list, calculate_probabilities: bool=True) -> np.array:
    """ Calculates the entropy for each feature in the given data. 

    :param data:                    Numpy representation of the data
    :param df_attrs:                Metadata for the given dataframe (important to distinguish between different column types)
    :param columns:                 Name of the columns
    :param calculate_probabilities: Whether or not to calculate probabilities or to directly use the values.

    :returns entropy for each feature
    """
    entropy_scores = []
    for i in tqdm(range(data.shape[1])):
        if df_attrs[columns[i]]["datatype"] == list:
            entropy_scores.append(single_column_entropy(data[:, i], True, calculate_probabilities))
        else:
            entropy_scores.append(single_column_entropy(data[:, i], False, calculate_probabilities))
        
    return np.array(entropy_scores)

def single_column_entropy(data: np.array, is_list: bool, calculate_probabilities: bool=True) -> np.float32:
    """ Calculates the entropy for a given feature.

    :param data:                    Numpy representation of the column
    :param is_list:                 Whether data is a list representation
    :param calculate_probabilities: Whether or not to calculate probabilities or to directly use the values.

    :returns entropy for the feature
    """
    if calculate_probabilities:
        if is_list:
            # Count internal occurences (e.g. flatten the data)
            _, counts = np.unique(utils.flatten_np_frozenset(data), return_counts=True)
        else:
            _, counts = np.unique(data, return_counts=True)
        counts = counts / data.shape[0]
    else:
        counts = data

    # Compute entropy
    return np.sum(-1 * np.log(counts) * counts)


def mutual_information_classic(data: np.ndarray, df_attrs: dict, columns: list):
    """ Calculates the mutual information between all features.

    :param data:                    Numpy representation of the column
    :param df_attrs:                Metadata for the given dataframe (important to distinguish between different column types)
    :param columns:                 Columns names for the dataset.

    :returns a #Features by #Features triangular matrix containing the MI values for each pair for feature.
    """

    # Create a feature_size x feature_size matrix
    mutual_information_matrix = np.zeros((data.shape[1], data.shape[1]))
    
    # List of marginal probabilities for each feature
    marginal_probabilities = []
    contingency_matrices = np.empty((data.shape[1], data.shape[1]), dtype=np.object_)
    lookUps = np.empty((data.shape[1], data.shape[1]), dtype=np.object_)
    # Mutual information is symmetric, so MI(f_i, f_j) == MI(f_j, f_i). Also MI(f_i, f_i) is always equal to 1. So no need for calculation.
    for i in tqdm(range(data.shape[1]-1)):
        for j in tqdm(range(i+1, data.shape[1]), leave=False):
            normalized_contingency_matrix, lookup_x, lookup_y = custom_contingency_matrix(data[:, i], data[:, j], \
            df_attrs[columns[i]]["datatype"] == list, df_attrs[columns[j]]["datatype"] == list)

            # Add the contingency matrix for later usage!
            contingency_matrices[i, j] = normalized_contingency_matrix
            lookUps[i, j] = (lookup_x, lookup_y)
            # Marginal_probabilities can directly be derived from the normalized contingency matrix, because it represents the joint probability distribution
            if len(marginal_probabilities) < i+1:
                marginal_probabilities.append(np.sum(normalized_contingency_matrix, axis=1))
            # Grab G(f_i)
            marginal_i = marginal_probabilities[i]

            if len(marginal_probabilities) < j+1:
                marginal_probabilities.append(np.sum(normalized_contingency_matrix, axis=0))

            # Grab G(f_j)
            marginal_j = marginal_probabilities[j]

            mutual_information = 0

            for i_dx in range(normalized_contingency_matrix.shape[0]):
                for j_dx in range(normalized_contingency_matrix.shape[1]):
                    if normalized_contingency_matrix[i_dx, j_dx] > 0.0:
                        term = normalized_contingency_matrix[i_dx, j_dx] * np.log(normalized_contingency_matrix[i_dx, j_dx]/(marginal_i[i_dx]*marginal_j[j_dx]))
                
                        mutual_information += term

            # Precision/Round Errors sometimes lead to small negative values in the e-5 to e-6 range, so filter this noise out
            mutual_information_matrix[i, j] = max(mutual_information, 0.)

    if config.STORE_IMAGES_DURING_EXECUTION or config.SHOW_IMAGES_DURING_EXECUTION:
        utils.check_dir_file(config.PATH_OUTPUT_IMAGES, False)
        path = f"{config.PATH_OUTPUT_IMAGES}/0_0_Unlabeled_Barplots.png"
        vis_utils.create_data_distribution_plot_unlabeled(columns, marginal_probabilities, df_attrs, config.STORE_IMAGES_DURING_EXECUTION, config.SHOW_IMAGES_DURING_EXECUTION, path)


        

    if config.STORE_IMAGES_DURING_EXECUTION or config.SHOW_IMAGES_DURING_EXECUTION:
        utils.check_dir_file(config.PATH_OUTPUT_IMAGES, False)
        path = f"{config.PATH_OUTPUT_IMAGES}/0_1_Unlabeled_Heatmaps_PLACE_X_PLACE_Y.png"
        vis_utils.create_pairplot_unlabeled(columns, contingency_matrices, marginal_probabilities, df_attrs, config.STORE_IMAGES_DURING_EXECUTION, config.SHOW_IMAGES_DURING_EXECUTION, path)


    return mutual_information_matrix


def calc_normalized_mi(mi_matrix: np.ndarray, entropy_values: np.array) -> np.ndarray:
    """ Calculates the normalized mutual information between all features.
        Ref: Vinh et al., 2010, using average entropy as upper limit to normalize mi matrix between [0, 1]

    :param mi_matrix:       Matrix containing MI values for each feature pair f_i != f_j
    :param entropy_values:  Entropy values for each feature

    :returns a normalized #Features by #Features triangular matrix containing the MI values for each pair for feature.
    """
    normalized_mi_matrix = np.zeros((mi_matrix.shape[0], mi_matrix.shape[1]))
    for i in range(mi_matrix.shape[0] - 1):
        for j in range(i+1, mi_matrix.shape[1]):
            normalized_mi_matrix[i, j] = (2 * mi_matrix[i, j]) / (entropy_values[i] + entropy_values[j])

    return normalized_mi_matrix

def fill_triangular_matrix(matrix):
    """ Mirrors the diagonal axis for the given matrix.
    
    :param matrix:  2D-triangular matrix

    :returns filled 2D-matrix
    """
    full_matrix = np.zeros((matrix.shape[0], matrix.shape[1]))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 0:
                full_matrix[i, j] = matrix[j, i]
            else:
                full_matrix[i, j] = matrix[i, j]

    return full_matrix

def calc_avg_norm_mi_scores(normalized_mi_matrix: np.ndarray, removed_indices: list = []) -> np.array:
    """ Average the normalized mutual information scores for each feature
    
    :param matrix:  Normalized filled MI Matrix

    :returns average NMI scores for each feature
    """
    # Divide by size-1, because one entry is always 0 (where f_i == f_i)
    return np.sum(normalized_mi_matrix, axis=1) / (normalized_mi_matrix.shape[0] - 1 - len(removed_indices))

def custom_contingency_matrix_one_list(x: np.array, y_list: np.array) -> np.ndarray:
    """ Calculate contingency matrix when one feature is a list feature

    :param x:       Feature column
    :param y_list:  Feature column (contains lists)

    :returns contingency matrix
    """

    unique_labels_x = np.unique(x)
    # Note that numpy returns the unique labels in a sorted manner, therefore i have the direct mapping through my df.attrs -> classnames
    unique_labels_y = np.unique(utils.flatten_np_frozenset(y_list))

    lookup_x = {value:index for index, value in enumerate(unique_labels_x)}
    lookup_y = {value:index for index, value in enumerate(unique_labels_y)}

    # Count through my own logic! (see custom_contingency_matrix_both_list function)
    result_counts = np.zeros((unique_labels_x.shape[0], unique_labels_y.shape[0]), dtype=np.float32)

    for x_value, y_value in zip(x, y_list):
        factor = 1 / len(y_value)
        for _y in y_value:
            result_counts[lookup_x[x_value], lookup_y[_y]] += factor
    

    return result_counts, lookup_x, lookup_y

def custom_contingency_matrix_both_list(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ Calculate contingency matrix when both features contain lists

    :param x:   Feature column (contains lists)
    :param y:   Feature column (contains lists)

    :returns contingency matrix
    """
    unique_labels_x = np.unique(utils.flatten_np_frozenset(x))
    unique_labels_y = np.unique(utils.flatten_np_frozenset(y))

    lookup_x = {value:index for index, value in enumerate(unique_labels_x)}
    lookup_y = {value:index for index, value in enumerate(unique_labels_y)}
    # Count through my own logic!
    result_counts = np.zeros((unique_labels_x.shape[0], unique_labels_y.shape[0]), dtype=np.float32)

    for x_value, y_value in zip(x, y):
        # One entriy in first, one entry in seconds, like x: [1], y: [4] -> count as 1 (one regular event)
        # Two entries in first, one entry in second, like x: [1,2], y: [5] -> count as 1/2 (two half weighted events)
        # Three entries in first, four entries in second, like: [1,2,3], y: [4,5,6,7] -> count as 1/12
        factor = 1 / (len(x_value) * len(y_value))
        for _x in x_value:
            for _y in y_value:
                result_counts[lookup_x[_x], lookup_y[_y]] += factor

    return result_counts, lookup_x, lookup_y



def custom_contingency_matrix(x: np.array, y: np.array, x_is_list: bool, y_is_list: bool) -> np.ndarray:
    """ Calculate contingency matrix

    :param x:   Feature column
    :param y:   Feature column
    :param x_is_list:   Whether column x contains lists
    :param y_is_list:   Whether column y contains lists
    
    :returns contingency matrix
    """
    # if x and y are numerical / categorical we can just return sklearn's implementation
    if x_is_list:
        if y_is_list:
            m, x_lookup, y_lookup = custom_contingency_matrix_both_list(x, y)
        else:
            m, y_lookup, x_lookup = custom_contingency_matrix_one_list(y, x)
            m = m.transpose()
    else:
        if y_is_list:
            m, x_lookup, y_lookup = custom_contingency_matrix_one_list(x, y)
        else:
            # Contingency matrix goes by sorted(unique_values(x)), same for y
            m = contingency_matrix(x, y)
            x_lookup = {value:index for index, value in enumerate(np.unique(x))}
            y_lookup = {value:index for index, value in enumerate(np.unique(y))}

    m = m / x.shape[0]   

    # Validate that the sum of marginal probabilities (which are represented by the sum of joint probabilities) are equal to 1 for both features         
    assert np.sum(np.sum(m, axis=0)) <= 1.001 and np.sum(np.sum(m, axis=0)) >= 0.999
    assert np.sum(np.sum(m, axis=1)) <= 1.001 and np.sum(np.sum(m, axis=1)) >= 0.999
    return m, x_lookup, y_lookup

def get_result_table(df: pd.DataFrame = pd.DataFrame()):
    """ Run Step 1 from the original paper with an adjusted ranking system. 

    :param df:  Dataframe containing the dataset and metainformation about the different colunms

    :returns ranked feature matrix
    """

    print(df)
    columns = list(df.columns)
    logging.info(f"\nThe following columns represent the data:\n{pd.DataFrame(columns, columns=['Column Names']).to_markdown()}")

    numpy_data = prepare_data_for_mutual_information(df) 
    logging.info(f"\nDataset Shape: {numpy_data.shape})")    
    
    entropy_values = feature_entropy(numpy_data, df.attrs, columns)
    print(f"Entropy Values for each feature:\n{entropy_values}\n")

    # For comparison between sklearn's MI implementation. Works only for integer values, therefore i compare my list-mi-variant.
    try:
        start = time.time()
        sklearn_matrix = np.zeros((numpy_data.shape[1], numpy_data.shape[1]))
        for i in range(numpy_data.shape[1]-1):
            for j in range(i+1, numpy_data.shape[1]):
                sklearn_matrix[i, j] = normalized_mutual_info_score(numpy_data[:, i], numpy_data[:, j])
        end= time.time()
        print(f"Execution for sklearn took {end-start} seconds")

        print(f"Sklearn Normalized Mutual Information matrix:\n{pd.DataFrame(sklearn_matrix)}\n")
    except Exception:
        print("Can't execute sklearn, most likely you used a dataset that contains lists.\n")
        
    start = time.time()
    mi_matrix = mutual_information_classic(numpy_data, df.attrs, columns)

    normalized_mi_matrix = calc_normalized_mi(mi_matrix, entropy_values)
    end = time.time()
    print(f"Execution for classic took {end-start} seconds")
    logging.info(f"\nNormalized Mutual Information Table, which took {end-start} seconds:\n\n{pd.DataFrame(normalized_mi_matrix).to_markdown()}")  
    
    # Not essential step, but makes the averaging alot easier (no dimensional logic to get all the correct values)
    normalized_filled_mi_matrix = fill_triangular_matrix(normalized_mi_matrix)
    print(f"Filled matrix:\n {pd.DataFrame(normalized_filled_mi_matrix)}\n")

    if config.STORE_IMAGES_DURING_EXECUTION or config.SHOW_IMAGES_DURING_EXECUTION:
        utils.check_dir_file(config.PATH_OUTPUT_IMAGES, False)
        path = f"{config.PATH_OUTPUT_IMAGES}/1_0_NMI_Heatmap.png"
        vis_utils.create_heatmap_nmi_scores(columns, normalized_filled_mi_matrix, config.STORE_IMAGES_DURING_EXECUTION, config.SHOW_IMAGES_DURING_EXECUTION, path)
   
    average_normalized_mi_scores = calc_avg_norm_mi_scores(normalized_filled_mi_matrix)

    if config.STORE_IMAGES_DURING_EXECUTION or config.SHOW_IMAGES_DURING_EXECUTION:
        utils.check_dir_file(config.PATH_OUTPUT_IMAGES, False)
        path = f"{config.PATH_OUTPUT_IMAGES}/1_1_Avg_NMI_Bar.png"
        indices = np.flip(np.argsort(average_normalized_mi_scores))
        columns_sorted = np.array(columns)[indices]
        x_labels = columns_sorted if not config.ANONIMIZE_FEATURE_NAMES else [str(i) for i in indices]
        vis_utils.create_avg_nmi_barchart(x_labels, average_normalized_mi_scores[indices], config.STORE_IMAGES_DURING_EXECUTION, config.SHOW_IMAGES_DURING_EXECUTION, path)

    copy_avg_normalized_mi_scores=deepcopy(average_normalized_mi_scores)
    worst_indices_attributes = []
    worst_avg_scores=[]
    worst_associated_partner = []
    size = normalized_filled_mi_matrix.shape[0]

    """ Updated feature ranking process. Essentially ranks based on highest remaining mutual-information.
    While True:
        - if len(MI-Table) <= 2: break
        - find the feature pair from MI-Table with the highest value
        - from the two features: find the feature with the higher average NMI (calculated from the MI-Table)
        - pop this feature (and add it to the ranking)
        - updated the MI-Table (by removing the column and row associated to the removed feature)
    """
    for i in range(size):
        checked_indices = np.argwhere(normalized_filled_mi_matrix>0)

        indices = np.unravel_index(normalized_filled_mi_matrix.argmax(), normalized_filled_mi_matrix.shape)
        if checked_indices.shape[0] <= 2:
            higher_index, lower_index = (indices[0], indices[1]) if copy_avg_normalized_mi_scores[indices[0]] > copy_avg_normalized_mi_scores[indices[1]] else(indices[1], indices[0])

            worst_indices_attributes.append(higher_index)
            worst_avg_scores.append(normalized_filled_mi_matrix[indices])
            worst_associated_partner.append(lower_index)

            worst_indices_attributes.append(lower_index)
            worst_avg_scores.append(normalized_filled_mi_matrix[indices])
            worst_associated_partner.append(higher_index)
            break


        indices = np.unravel_index(normalized_filled_mi_matrix.argmax(), normalized_filled_mi_matrix.shape)

        if indices[0] == indices[1]:
            raise RuntimeError("For some reason there was a diagonal value picked from the filled NMI matrix")
        option_a = indices[0]
        option_b = indices[1]

        # Check for the option with the higher avg score
        avg_1 = average_normalized_mi_scores[option_a]
        avg_2 = average_normalized_mi_scores[option_b]

        if avg_1 > avg_2:
            target=option_a
        else:
            target=option_b 

        worst_avg_scores.append(normalized_filled_mi_matrix[indices])
        # Fill matrix with really high values
        normalized_filled_mi_matrix[:, target] = np.array([0]*size)
        normalized_filled_mi_matrix[target,:] = np.array([0]*size)  
        
        worst_indices_attributes.append(target)
        worst_associated_partner.append(option_a if target != option_a else option_b)
        # Recalculate averages without the removed attribute
        average_normalized_mi_scores = calc_avg_norm_mi_scores(normalized_filled_mi_matrix, worst_indices_attributes)

    worst_indices_attributes.reverse()
    worst_avg_scores.reverse()
    worst_associated_partner.reverse()
    feature_rank_indices = worst_indices_attributes
    
    ranked_feature_set = entropy_values - copy_avg_normalized_mi_scores

    # Paper suggests dropping each feature with a ranking score that is lower than 1. Doesn't make much sense though.
    old_feature_rank_indices = np.argsort(ranked_feature_set)[::-1]

    # Experiments on which attributes to remove ()
    old_result_view_step_1 = pd.DataFrame(np.array([[columns[index], np.around(entropy_values[index], 6), np.around(copy_avg_normalized_mi_scores[index], 6), np.around(ranked_feature_set[index], 6)]
    for index in old_feature_rank_indices if ranked_feature_set[index] != 0]), columns=['Feature Name', 'Entropy', 'Avg Norm MI Score', 'Old final Ranking'])
    
    new_result_view_step_1 = pd.DataFrame((
    np.array([[columns[index],
    np.around(entropy_values[index], 6),
    np.around(copy_avg_normalized_mi_scores[index], 6),
    np.around(worst_avg_scores[i], 6),
    columns[worst_associated_partner[i]]]
    for i, index in enumerate(feature_rank_indices)])), columns=['Feature Name', 'Entropy', 'Total Avg NMI Score', 'Highest Remaining NMI', 'Associated Partner'])

    old_prettified_view = old_result_view_step_1.set_index('Feature Name', inplace=False)
    
    prettified_view = new_result_view_step_1.set_index('Feature Name', inplace=False)
    
    logging.info(f"\nOld Result Table - showcasing Entropy, Avg Norm MI Score and the Old Final Ranking:\n\n{old_prettified_view.to_markdown()}")
    
    logging.info(f"\nResult Table - showcasing Entropy, Avg Norm MI Score and the Highest (remaining) NMI:\n\n{prettified_view.to_markdown()}")
    
    return new_result_view_step_1, old_result_view_step_1

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    dataset = CustomDataset(config.create_processed_data_path(True))

    res = get_result_table(dataset.get_training_df())