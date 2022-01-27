import sys
from pathlib import Path
from sklearn.metrics import cluster
sys.path.insert(0, str(Path('.').absolute()))

import matplotlib.pyplot as plt
plt.set_loglevel('WARNING') 
import umap.umap_ as umap
from tqdm import tqdm
import math

import numpy as np
import matplotlib.patches as mpatches
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import pandas as pd
import numbers

import config


def _transform_value_to_label(label_list, value):
    """ Transform a given value to a displayable label.
    
    :param label_list:      List of available labels for the given feature
    :param value:           Value to look up in the label_list and transform

    :returns transformed label representation of the given value
    """
    if isinstance(value, frozenset):
        if config.ANONIMIZE_FEATURE_NAMES:
            return {f"{v}" for v in value}
        else:    
            return {label_list[v] for v in value}
    elif isinstance(value, float) or isinstance(value, np.float32):
        return value

    else:
        if len(label_list) <= value:
            return value
        
        if config.ANONIMIZE_FEATURE_NAMES:
            return value
        return label_list[value]

def transform_feature_names(feature_names):
    """ Transform a list of feature names to a readable short-format. Potentially anonymizes. 
    :param feature_names:       List of strings to transform

    :returns transformed list of feature names
    """
    if len(feature_names) == 0:
        return feature_names
    if isinstance(feature_names[0], numbers.Number):
        return feature_names
    if config.ANONIMIZE_FEATURE_NAMES:
        return [f"{i}" for i, f in enumerate(feature_names)]
    else:
        return [feature_name[:3*config.MAX_CHARACTERS_TO_DISPLAY] for feature_name in feature_names]

def create_barchart(x_labels, y_values, y_title, x_title, plot_title, ax, alpha=1, cluster_index=0, index=""):
    """ Creates a barchart from the given data. Mostly taken from ref: https://www.pythoncharts.com/matplotlib/beautiful-bar-charts-matplotlib/

    :param x_labels:        Labels for the x-Axis
    :param y_values:        Values for the bars
    :param y_title:         Title of the Y-Axis
    :param x_title:         Title of the X-Axis
    :param plot_title:      Title of the whole plot
    :param ax:              Optional axis to write to
    :param alpha:           Transparency value for the bars
    :param cluster_index:   Index of the current cluster to determine the color of the bars
    :param index:           Index of the current feature
    
    """
    color = sns.color_palette()[cluster_index]
    if len(x_labels) == 0:
        x_labels = [str(value) for value in np.arange(len(y_values))]
        bars = ax.bar(
            x= x_labels,
            height=y_values,
            alpha=alpha
        )

    else:
        x_labels = transform_feature_names(x_labels)
        bars = ax.bar(
        x= x_labels,
        height=y_values,
        color=color,
        alpha=alpha

    )

    # Axis formatting.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)

    if len(y_values) < 10:
        digits=3
    elif len(y_values) > 16:
        digits=0
    else:
        digits=2
    # Add text annotations to the top of the bars.
    value_treshold = 0.000
    if digits > 0:
        bar_color = bars[0].get_facecolor()
        for bar in bars:
            if bar.get_height() > value_treshold:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height()+.006,
                    round(bar.get_height(), digits),
                    horizontalalignment='center',
                    color=bar_color,
                    weight='bold'
                )

    ax.set_xticks(x_labels)
    ax.set_xticklabels([x_label if y_values[i]>= value_treshold else "" for i, x_label in enumerate(x_labels) ])

    if len(x_labels) > 50:
        ax.set_xticklabels([])
    
    if not isinstance(x_labels[0], numbers.Number) and not config.ANONIMIZE_FEATURE_NAMES:
        for tick in ax.get_xticklabels():
            tick.set_rotation(20)
    # Add labels and a title.
    if len(x_title) > 0:
        if config.ANONIMIZE_FEATURE_NAMES:
            x_title = f"Feature {index}"
        ax.set_xlabel(x_title, labelpad=15, color='#333333')
    if len(y_title) > 0:
        ax.set_ylabel(y_title, labelpad=15, color='#333333')
    if len(plot_title) > 0:
        if config.ANONIMIZE_FEATURE_NAMES:
            ax.set_title(plot_title, pad=15, color='#333333', weight='bold')



def create_avg_nmi_barchart(x_labels, y_values, store_image, show_image, path):
    """ Creates the barchart for the avg normalized mutual information values.
    
    :param x_labels:    Labels for the features
    :param y_values:    Avg NMI Values
    :param store_image: Whether to store the image or not
    :param show_image:  Whether to show the image during execution or not
    :param path:        Path to store the image to
    """
    fig, ax = plt.subplots()
    create_barchart(x_labels, y_values, "Average NMI", "Feature", "", ax)
    ax.set_xticklabels(x_labels)
    fig.set_size_inches(*config.DISPLAY_SIZE)

    if store_image:
        plt.savefig(path, dpi=config.DPI_TO_STORE)
    if show_image:
        plt.show()



def create_heatmap_nmi_scores(feature_names, nmi_scores, store_image, show_image, path):
    """ Creates a heatmap for each normalized mutual information score between unique feature pairs

    :param feature_names:   Names of the features
    :param nmi_scores:      2D Numpy array of NMI Scores
    :param store_image:     Whether to store the image or not
    :param show_image:      Whether to show the image during execution or not
    :param path:            Path to store the image to
    
    """
    if not store_image and not show_image:
        return
    feature_names = transform_feature_names(feature_names)

    mask = np.zeros_like(nmi_scores, dtype=np.bool_)
    mask[np.triu_indices_from(mask)] = True

    _create_heatmap_plot_masked(None, feature_names, feature_names, nmi_scores, mask=mask)
    plt.xlabel("Feature", weight='bold')
    plt.ylabel("Feature", weight='bold')

    if store_image:
        plt.savefig(path, bbox_inches='tight', dpi=config.DPI_TO_STORE)
    if show_image:
        plt.show()

def create_data_distribution_plot_unlabeled(feature_names, marginal_probabilities, df_attrs, store_image, show_image, path):
    """ Creates a general data distribution plot of the given data.

    :param feature_names:           The names of the features
    :param marginal_probabilities:  Marginal probabilities for each unique value for each feature
    :param df_attrs:                Meta information for the given dataframe
    :param store_image:             Whether to store the image or not
    :param show_image:              Whether to show the image during execution or not
    :param path:                    Path to store the image to 
    """
    
    if not store_image and not show_image:
        return
    n_features = len(marginal_probabilities)
    rows = math.ceil(math.sqrt(n_features))
    cols = math.ceil(n_features/rows)

    fig, axs = plt.subplots(cols, rows)
    for i, ax in enumerate(axs.flat):
        if i >= n_features:
            for k in range (i, len(axs.flat)):
                fig.delaxes(axs.flat[k])
            break
   
        value_labels_i = df_attrs[feature_names[i]].get("classes", [])
        datatype = str(df_attrs[feature_names[i]].get("datatype"))
        create_barchart(value_labels_i, marginal_probabilities[i], "", feature_names[i], datatype, ax, alpha=1.0, index=i)

    fig.set_size_inches(*config.DISPLAY_SIZE)
    fig.tight_layout()
    if store_image:
        plt.savefig(path, bbox_inches='tight', dpi=config.DPI_TO_STORE)
    if show_image:
        plt.show()

    plt.close()

def create_pairplot_unlabeled(feature_names, contingency_matrices, marginal_probabilities, df_attrs, store_image, show_image, path):
    """ Create pairplot for unlabeled data
    Format for marginal_probabilities: [{"labels": ["American", "Asian", "Greek"], "marginal_probabilities": [0.1, 0.2, 0.7]}, {}, ..]
    
    :param feature_names            List of feature names
    :param contingency_matrices     Holds all contingency matrices for all unique feature pairs
    :param marginal_probabilities   Holds all the marginal probabilities for each feature
    :param df_attrs                 Dict, description on the given columns
    :param store_image              Boolean, whether to store the image to disk or not
    :param show_image               Boolean, whether to show the image during execution
    :param path                     String, path to potentially store the image to
"""
    if not store_image and not show_image:
        return

    n_features = len(marginal_probabilities)
    
    for i in range(n_features):
        for j in range(i+1, n_features):
            value_labels_i = df_attrs[feature_names[i]].get("classes", [])
            value_labels_j = df_attrs[feature_names[j]].get("classes", [])

            display_i = str(i) if config.ANONIMIZE_FEATURE_NAMES else feature_names[i]
            display_j = str(j) if config.ANONIMIZE_FEATURE_NAMES else feature_names[j]
            if contingency_matrices[i, j].shape[0] >= 30 or contingency_matrices[i, j].shape[1] >= 30:
                print(f"[WARN] Skipping feature-pair {display_i}-{display_j} because there are too many unique values: {contingency_matrices[i, j].shape[0]}-{contingency_matrices[i, j].shape[1]}")
                plt.close()
                continue

            _create_heatmap_plot(None, value_labels_i, value_labels_j, np.transpose(contingency_matrices[i, j]), True, True, scale_prob=True)
            plt.xlabel(f"Feature {display_i}")
            plt.ylabel(f"Feature {display_j}")

            if config.ANONIMIZE_FEATURE_NAMES:
                current_path = path.replace("PLACE_X", str(i))
                current_path = current_path.replace("PLACE_Y", str(j))
            else:
                current_path = path.replace("PLACE_X", feature_names[i])
                current_path = current_path.replace("PLACE_Y", feature_names[j])
            if store_image:
                plt.savefig(current_path, bbox_inches='tight', dpi=config.DPI_TO_STORE)
            if show_image:
                plt.show()

            plt.close()

def create_linecharts_elbow_method(x_values, costs, ch_indices, store_image, show_image, path):
    """ Simple multiplot from the output of the elbow-method function.
    
    :param x_values:                Labels for x-Values
    :param costs:                   List of Costs
    :param ch_indices:              List of CH-Indices values
    :param store_image              Boolean, whether to store the image to disk or not
    :param show_image               Boolean, whether to show the image during execution
    :param path                     String, path to potentially store the image to
    """
    fig, ax = plt.subplots()

    ax.plot(x_values, costs, color="r", label="Adjusted Costs")
    ax.set_xlabel("Number of Clusters k") 
    ax.set_ylabel("Total Costs", color="red")

    ax2 = ax.twinx()
    ax2.plot(x_values, ch_indices, color="blue", label="Custom CH-Index")
    ax2.set_ylabel("CH-Index", color="blue")

    plt.tight_layout()
    fig.set_size_inches(*config.DISPLAY_SIZE)
    if store_image:
        plt.savefig(path, bbox_inches='tight', dpi=config.DPI_TO_STORE)
    if show_image:
        plt.show()

def create_cluster_table(indices_map, columns, centroids, labels, attrs):
    """ Returns a table that summarized the information for the given centroids

    :param indices_map  Used indices from the dataset
    :param columns      List of columns
    :param centroids    Centroids for each cluster
    :param labels       Labels for each cluster
    :param attrs        Metadata of the given dataframe
    
    :returns String representation for both tables
    """

    feature_names = []
    ordered_feature_names = []
    columns = np.array(columns)
    
    for i, col in enumerate(columns):
        for name, indices in indices_map.items():
            if name == "time_max_values":
                continue
            if i in indices:
                ordered_feature_names.append(col)
                break

    
    for name, indices in indices_map.items():
        if name == "time_max_values":
            continue
        feature_names.append(columns[indices])
    

    # Flatten list
    feature_names = [item for sublist in feature_names for item in sublist]
    clusters = []
    for _ in range(centroids[1].shape[0]):
        clusters.append(list())

    for feature_list in centroids:
        for i, centroid_features in enumerate(feature_list):
            if len(centroid_features)>0:
                clusters[i].append(centroid_features)

    for i in range(len(clusters)):
        clusters[i] = [item for sublist in clusters[i] for item in sublist]


    for i, feature_name in enumerate(feature_names):
        for j in range(len(clusters)):
            clusters[j][i] = _transform_value_to_label(attrs[feature_name].get("classes", []), clusters[j][i])


    df_centroids = pd.DataFrame(clusters, columns = feature_names)
    # Reorder columns:
    df_centroids = df_centroids[ordered_feature_names]

    df_centroids.index = [i for i in range(0, len(clusters))]
    df_centroids.index.name = 'Cluster'
    
    if config.ANONIMIZE_FEATURE_NAMES:
        df_centroids.columns = [f"Feature {list(columns).index(i)}" for i in ordered_feature_names]
    # Use counts to create the table
    unique_values, counts = np.unique(labels, return_counts=True)
    df_counts =pd.DataFrame(list(zip(unique_values, counts)), columns=["Cluster", "Number of Points"])
    df_counts.set_index('Cluster', inplace=True)

    return df_centroids, df_counts
    
def visualize_labeled_dataframe_transformation(data_frame: pd.DataFrame, target_labels: list, k: int, indices_map: dict, samples_per_cluster, store_image, show_image):
    """ Creates Barplots and Heatmaps for each cluster of data.

    :param data_frame:              The given dataframe
    :param target_labels:           The labels for the data from the cluster run
    :param k:                       The number of clusters
    :param indices_map:             The currently used indices per datatype
    :param samples_per_cluster:     Datapoint counts per cluster
    :param store_image              Boolean, whether to store the image to disk or not
    :param show_image               Boolean, whether to show the image during execution
    """
    # Necesarry to not have cross imports
    from src.logic_part_1 import custom_contingency_matrix, prepare_data_for_mutual_information

    if not store_image and not show_image:
        return

    samples_per_cluster = np.squeeze(samples_per_cluster.values)
    feature_indices = []
    columns = np.array(data_frame.columns)
    for name, indices in indices_map.items():
        if name == "time_max_values":
            continue
        feature_indices.append(indices)
    # Flatten list
    feature_indices = sorted([item for sublist in feature_indices for item in sublist])

    n_features = len(feature_indices)
    contingency_matrices_per_cluster = np.empty((n_features, n_features, k), dtype=np.object_)
    marginal_probabilities_per_cluster = np.empty((n_features, k), dtype=np.object_)

    look_ups = np.empty((n_features, k), dtype=np.object_)

    # For each cluster
    print(f"Starting to calculate Joint and Marginal Probabilities")
    X = prepare_data_for_mutual_information(data_frame)
    for cluster_index in tqdm(range(k)):
        associated_points = np.argwhere(cluster_index == target_labels).flatten()
        data_cluster = X[associated_points]

        # For each pair of feature i!=j
        for i in tqdm(range(len(feature_indices)), leave=False):
            for j in tqdm(range(i, len(feature_indices)), leave=False):
                if i == j:
                    # Calculate data distribution for the current cluster
                    continue
                # Feature Index I
                index_i = feature_indices[i]

                # Feature Index J
                index_j = feature_indices[j]

                feature_name_i = columns[index_i]
                feature_name_j = columns[index_j]
                # Calculate normalized
                # Lookup should be a map, with Value -> Index lookup! (e..g lookup_x.keys() gives all unique values, lookup_x.values() gives associated )

                normalized_contingency_matrix, lookup_x, lookup_y = custom_contingency_matrix(data_cluster[:, index_i], data_cluster[:, index_j], \
                data_frame.attrs[columns[index_i]]["datatype"] == list, data_frame.attrs[columns[index_j]]["datatype"] == list)

                # Just in case
                temp = look_ups[i, cluster_index]
                if not isinstance(look_ups[i, cluster_index], dict):
                    look_ups[i, cluster_index] = lookup_x
                if not isinstance(look_ups[j, cluster_index], dict):
                    look_ups[j, cluster_index] = lookup_y

                contingency_matrices_per_cluster[i, j, cluster_index] = normalized_contingency_matrix
                
                contingency_matrices_per_cluster[j, i, cluster_index] = normalized_contingency_matrix.transpose()
                

                if not isinstance(marginal_probabilities_per_cluster[i, cluster_index], np.ndarray):
                    marginal_probabilities_per_cluster[i, cluster_index] = np.sum(normalized_contingency_matrix, axis=1)

                if not isinstance(marginal_probabilities_per_cluster[j, cluster_index], np.ndarray):
                    marginal_probabilities_per_cluster[j, cluster_index] = np.sum(normalized_contingency_matrix, axis=0)

    max_cols = config.MAX_COLS_TO_DISPLAY
    rows = k

    for temp in range(math.ceil(n_features/max_cols)):
        # From temp to temp+cols
        considerable_features = [i for i in range(max_cols*temp, min(max_cols*(temp+1), n_features))]
        fig, axs = plt.subplots(rows, len(considerable_features))

        for cluster_index in range(rows):
            for idx, j in enumerate(considerable_features):
                feature_index = feature_indices[j]
                feature_name = columns[feature_index]
                if len(considerable_features) > 1:
                    current_axs = axs[cluster_index, idx]
                else:
                    current_axs = axs[cluster_index]

                value_labels_i = data_frame.attrs[feature_name].get("classes", [])      
                if len(value_labels_i) == 0:
                    max_unique_values = data_frame.attrs[feature_name].get("max_time_value", data_frame.attrs[feature_name].get("number_of_bins", 0))
                    value_labels_i = [v for v in range(max_unique_values)]

                current_look_ups = look_ups[j, cluster_index]
                current_marginal_probs = marginal_probabilities_per_cluster[j, cluster_index]
                assert len(current_marginal_probs) == len(current_look_ups), "weird error"
                full_marginal_probabilities = []
                for value_label_index in range(len(value_labels_i)):
                    index = current_look_ups.get(value_label_index, "-1")
                    
                    if index != "-1":
                        full_marginal_probabilities.append(current_marginal_probs[index])
                    else:
                        full_marginal_probabilities.append(0.0)

                create_barchart(value_labels_i, full_marginal_probabilities, "", feature_name + f", n={samples_per_cluster[cluster_index]}", "", current_axs, alpha=1.0, cluster_index=cluster_index, index=feature_index)
        
        fig.set_size_inches(*config.DISPLAY_SIZE)
        fig.tight_layout()
        all_current_features = []
        for x in considerable_features:
            feature_index = feature_indices[x]
            if config.ANONIMIZE_FEATURE_NAMES:
                all_current_features.append(feature_index)
            else:
                all_current_features.append(columns[feature_index])

        path = f"{config.PATH_OUTPUT_IMAGES}/4_0_Labeled_Barplots_{all_current_features}.png"
        
        if store_image:
            plt.savefig(path, bbox_inches='tight', dpi=config.DPI_TO_STORE)
        if show_image:
            plt.show()

        plt.close(fig)

    for i in range(n_features):
        for j in range(i+1, n_features):
            feature_index_i = feature_indices[i]
            feature_index_j = feature_indices[j]
            plottable_feature_pair = True
            feature_name_i = columns[feature_index_i]
            feature_name_j = columns[feature_index_j]

            fig, axs = plt.subplots(rows, 1) 
            for cluster_index in range(k):

                current_axs = axs[cluster_index]
                value_labels_i = data_frame.attrs[feature_name_i].get("classes", [])
                value_labels_j = data_frame.attrs[feature_name_j].get("classes", [])
                
                if len(value_labels_i) == 0:
                    max_unique_values = data_frame.attrs[feature_name_i].get("max_time_value", data_frame.attrs[feature_name_i].get("number_of_bins", 0))
                    value_labels_i = [v for v in range(max_unique_values)]
                
                if len(value_labels_j) == 0:
                    max_unique_values = data_frame.attrs[feature_name_j].get("max_time_value", data_frame.attrs[feature_name_j].get("number_of_bins", 0))
                    value_labels_j = [v for v in range(max_unique_values)]
                
                if len(value_labels_i) >= 30 or len(value_labels_j) >= 30:
                    print(f"[WARN] Skipping feature-pair {feature_name_i}-{feature_name_j} because there are too many unique values: {len(value_labels_i)}-{len(value_labels_j)}")
                    plt.close(fig)
                    plottable_feature_pair = False
                    break

                if columns[feature_index_i] == "unixTimestamp" or columns[feature_index_j] == "unixTimestamp":
                    print(f"[INFO] Skipping unixTimestamp")
                    plt.close(fig)
                    plottable_feature_pair = False
                    break
                
                current_look_ups_i = look_ups[i, cluster_index]
                current_look_ups_j = look_ups[j, cluster_index]

                # Problem of filling the zeros?
                joint_probabilities = contingency_matrices_per_cluster[i, j, cluster_index]

                full_joint_probabilities = np.zeros((len(value_labels_i), len(value_labels_j)), dtype=np.float64)

                for value_label_index_i in range(len(value_labels_i)):
                    for value_label_index_j in range(len(value_labels_j)):
                        index_i = current_look_ups_i.get(value_label_index_i, "-1")
                        index_j = current_look_ups_j.get(value_label_index_j, "-1")
                        if index_i != "-1" and index_j != "-1":
                            full_joint_probabilities[value_label_index_i, value_label_index_j] = joint_probabilities[index_i, index_j]
                        else:
                            full_joint_probabilities[value_label_index_i, value_label_index_j] = 0.0
               
                _create_heatmap_plot(axs[cluster_index], value_labels_j, value_labels_i, full_joint_probabilities, True, True, False, True)

                if config.ANONIMIZE_FEATURE_NAMES:
                    axs[cluster_index].set_ylabel(f"Feature {feature_index_i}", weight='bold')
                    axs[cluster_index].set_xlabel(f"Feature {feature_index_j}", weight='bold')
                else:
                    axs[cluster_index].set_ylabel(f"{feature_name_i}", weight='bold')
                    axs[cluster_index].set_xlabel(f"{feature_name_j}", weight='bold')
        
            if not plottable_feature_pair:
                continue

            fig.set_size_inches(*config.DISPLAY_SIZE)
            fig.tight_layout()
            
            if config.ANONIMIZE_FEATURE_NAMES:
                path = f"{config.PATH_OUTPUT_IMAGES}/4_1_Labeled_Heatmaps_{feature_index_i}_{feature_index_j}.png"
            else:
                path = f"{config.PATH_OUTPUT_IMAGES}/4_1_Labeled_Heatmaps_{feature_name_i}_{feature_name_j}.png"

            if store_image:
                plt.savefig(path, bbox_inches='tight', dpi=config.DPI_TO_STORE)
            if show_image:
                plt.show()

            plt.close(fig)

def create_umap_plot(data_frame: pd.DataFrame, target_labels: list, k: int, indices_map: dict, store_image, show_image, path, dist_fct=config.UMAP_PARAM_METRIC, n_neighbors=config.UMAP_PARAM_N_NEIGHBORS, min_dist=config.UMAP_PARAM_MIN_DIST):
    """ Creates a UMAP plot for the given data and labels.

    :param data_frame:              The given dataframe
    :param target_labels:           The labels for the data from the cluster run
    :param k:                       The number of clusters
    :param indices_map:             The currently used indices per datatype
    :param store_image              Boolean, whether to store the image to disk or not
    :param show_image               Boolean, whether to show the image during execution
    :param path:                    Path to store the plots in
    :param dist_fct:                Distance metric to be used by UMap
    :param n_neighbors:             Number of Neighbors to be used by UMap
    :param min_dist:                Minimum distance to be used by UMap
    """
    
    all_indices = []
    for key, v in indices_map.items():
        
        if key != "time_max_values":
            all_indices = all_indices + v

    X = np.empty((len(data_frame.index), len(all_indices)), dtype=np.object_)

    last_index = 0

    _, counts = np.unique(target_labels, return_counts=True)
    print(f"Label counts: {counts}")

    data = data_frame.values
    categorical_indices = indices_map.get("cat_indices")
    time_indices=indices_map.get("time_indices")
    list_indices=indices_map.get("list_indices")
    numerical_indices = indices_map.get("num_indices")

    for i, index in enumerate(categorical_indices+time_indices):
        X[:, i] = data[:, index].astype(str)

    last_index += len(categorical_indices)+len(time_indices)

    for i, index in enumerate(list_indices):
        X[:, i+last_index] = LabelEncoder().fit_transform(data[:, index]).astype(str)

    last_index += len(list_indices)


    for i, index in enumerate(numerical_indices):
        X[:, i+last_index] = data[:, index]

    vis = umap.UMAP(verbose=True, n_jobs=12, n_neighbors=n_neighbors, metric=dist_fct, low_memory=False, min_dist=min_dist)
    embedding = vis.fit_transform(X, target_labels)

    fig, ax = plt.subplots()
    plt.scatter(embedding[:, 0], embedding[:, 1], c=[sns.color_palette()[x] for x in target_labels])
    plt.gca().set_aspect('equal', 'datalim')

    ax.grid(True)
    plt.legend(handles=[mpatches.Patch(color=sns.color_palette()[i], label=i) for i in range(k)])

    fig.set_size_inches(*config.DISPLAY_SIZE)
    if store_image:
        plt.savefig(path, bbox_inches='tight', dpi=config.DPI_TO_STORE)
    if show_image:
        plt.show()

def _create_heatmap_plot(ax, x_labels, y_labels, contincengy_matrix: np.ndarray, show_x:bool, show_y:bool, show_values:bool = False, scale_prob:bool = False):
    """ Create heatmap plots for each pair of features, where feature_i != feature_j
    
    :param ax:                  Current matplotlib ax to write the plot to
    :param x_labels:            List of Strings representing the x-Labels
    :param y_labels:            List of Strings representing the y-Labels
    :param contingency_matrix:  2D np array of joint probabilities
    :param show_x:              Boolean, whether to display the x-labels
    :param show_y:              Boolean, whether to display the y-labels
    :param show_values:         Boolean, whether to display the values in the heatmap
    :param scale_prob:          Boolean, whether to scale colors to the max value in the contingency matrix or to 1
    """

    x_labels_transformed = transform_feature_names(x_labels if len(x_labels) > 0 else [i for i in range(contincengy_matrix.shape[1])])
    y_labels_transformed = transform_feature_names(y_labels if len(y_labels) > 0 else [i for i in range(contincengy_matrix.shape[0])])

    if not show_x:
        x_labels_transformed = False
    if not show_y:
        y_labels_transformed= False
    if scale_prob:
        vmax = min(1, contincengy_matrix.max())
    else:
        vmax = 1

    ax = sns.heatmap(contincengy_matrix, vmin=0, vmax=vmax, cmap="magma_r", annot=show_values, fmt=".3f", ax=ax, xticklabels=x_labels_transformed, yticklabels=y_labels_transformed, linewidths=0.1, linecolor="black")
    ax.plot()

    return ax

def _create_heatmap_plot_masked(ax, x_labels, y_labels, contincengy_matrix: np.ndarray, mask:np.array = np.empty(0)):
    """ Create heatmap plots for each pair of features, where feature_i != feature_j for a given mask and adjusted settings
    
    :param ax:                  Current matplotlib ax to write the plot to
    :param x_labels:            List of Strings representing the x-Labels
    :param y_labels:            List of Strings representing the y-Labels
    :param contingency_matrix:  2D np array of joint probabilities
    :param mask:                2D np array determining which values to show
    """

    x_labels_transformed = transform_feature_names(x_labels if len(x_labels) > 0 else [i for i in range(contincengy_matrix.shape[1])])
    y_labels_transformed = transform_feature_names(y_labels if len(y_labels) > 0 else [i for i in range(contincengy_matrix.shape[0])])
    if mask.size == 0:
        mask = np.zeros((len(y_labels_transformed), len(x_labels_transformed)), dtype=np.bool_)
    vmax = min(1, contincengy_matrix.max())
    with sns.axes_style("white"):
        ax = sns.heatmap(contincengy_matrix, vmin=0, vmax=vmax, cmap="magma_r", annot=False, ax=ax, xticklabels=x_labels_transformed, yticklabels=y_labels_transformed, mask=mask, square=True)
    
    ax.plot()

    return ax