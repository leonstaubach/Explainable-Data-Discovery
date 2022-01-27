import logging
from multiprocessing import cpu_count
import sys

import config
import src.utils as utils
from src.logic_part_1 import get_result_table
from src.logic_part_2 import execute_iterative_clustering
from src.data_loader import CustomDataset
from src.visualisation_utils import create_umap_plot, create_cluster_table, visualize_labeled_dataframe_transformation
from src.metrics import create_explicit_feature_importance
from src.kmodes.kprototypes import KPrototypes
from src.kmodes.util.dissim import matching_dissim, euclidean_dissim, matching_dissim_lists, time_dissim

def main():
    logging.info('\n' + '-' * 15 + ' BEGINNING NEW PROCESS'  + '-' * 15)
    logging.info(utils.config_to_str())

    # Analyse relevant features from result_table
    utils.check_dir_file(config.PATH_OUTPUT_METADATA, True)
    meta_data = utils.load_meta_data_attribute(config.PATH_OUTPUT_METADATA)
    
    # 1. Load Data
    dataset = CustomDataset(config.create_processed_data_path(True))
    
    if not config.TRY_PRELOAD_FEATURES or "feature_ranking" not in meta_data or "old_feature_ranking" not in meta_data:
        # 2. Get Entropy & Mutual Information, create general data distribution plots
        result_table, old_result_table = get_result_table(dataset.get_training_df())

        # Write results to table
        ranked_features = list(result_table['Feature Name'])
        old_ranked_features = list(old_result_table['Feature Name'])
        meta_data["feature_ranking"] = ranked_features
        utils.update_meta_data_attribute(f"{config.PATH_OUTPUT_METADATA}", "feature_ranking", ranked_features)
        utils.update_meta_data_attribute(f"{config.PATH_OUTPUT_METADATA}", "old_feature_ranking", old_ranked_features)

    # Update training df here based on the remaining feature ranks (keep names in order)
    dataset.update_training_df(sorted(meta_data.get("feature_ranking")))

    optimal_indices = None
    if config.TRY_PRELOAD_FEATURES:
        optimal_indices = meta_data.get("best_indices", None)

    # 3. Start process to remove features iteratively
    if not optimal_indices:
        optimal_indices = execute_iterative_clustering(dataset.get_training_df(), meta_data)
        meta_data["best_indices"] = optimal_indices
        utils.update_meta_data_attribute(f"{config.PATH_OUTPUT_METADATA}", "best_indices", optimal_indices)


    # 4. Apply clustering
    n_jobs = min(cpu_count(), config.K_PROTOTYPE_REPEAT_NUM)
    verbose = False if n_jobs > 1 else True
    k = meta_data["k"]

    logging.info(f"\nStarting K-Prototypes with optimal feature set: {optimal_indices} ..")        
    optimal_k_prototype = KPrototypes(n_clusters=k, init='Cao', n_init=config.K_PROTOTYPE_REPEAT_NUM, verbose=verbose, n_jobs=n_jobs)
    optimal_prototype_labels, _, _ = optimal_k_prototype.fit_predict(dataset.get_training_df(), indices_map=optimal_indices)

    utils.check_dir_file(config.PATH_OUTPUT_IMAGES, False)

    # 5. Create comparable UMAP view between the full and the optimal feature set
    if config.STORE_IMAGES_DURING_EXECUTION or config.SHOW_IMAGES_DURING_EXECUTION:
        path = f"{config.PATH_OUTPUT_IMAGES}/3_0_FULL_FEATURE_k={k}.png"
        full_indices_map = utils.create_indices_map(dataset.get_training_df().columns, dataset.get_training_df().attrs)

        logging.info(f"\nStarting K-Prototypes with {n_jobs} threads to create initial UMap visualisation ..")
        full_k_prototype = KPrototypes(n_clusters=k, init='Cao', n_init=config.K_PROTOTYPE_REPEAT_NUM, verbose=verbose, n_jobs=n_jobs)
        full_prototype_labels, _, _ = full_k_prototype.fit_predict(dataset.get_training_df(), indices_map=full_indices_map)
        create_umap_plot(dataset.get_training_df(), full_prototype_labels, k, full_indices_map, config.STORE_IMAGES_DURING_EXECUTION, config.SHOW_IMAGES_DURING_EXECUTION, path)
        
        path = f"{config.PATH_OUTPUT_IMAGES}/3_1_OPTIMAL_FEATURE_k={k}.png"
        create_umap_plot(dataset.get_training_df(), optimal_prototype_labels, k, optimal_indices, config.STORE_IMAGES_DURING_EXECUTION, config.SHOW_IMAGES_DURING_EXECUTION, path)
    
    # 6. Create tables to represeent the centroids
    centroids_table, counts_table = create_cluster_table(optimal_indices, list(dataset.get_training_df().columns), optimal_k_prototype.cluster_centroids_, optimal_k_prototype.labels_, dataset.get_training_df().attrs)
    logging.info(f"\nFinal Centroids:\n{centroids_table.to_markdown()}\n\nWith the following data point counts:\n{counts_table.to_markdown()}")

    # 7. Create a feature importance ranking
    distance_functions = [euclidean_dissim, matching_dissim, matching_dissim_lists, time_dissim]
    centroids_feature_map = create_explicit_feature_importance(optimal_indices, optimal_prototype_labels, list(dataset.get_training_df().columns), optimal_k_prototype.cluster_centroids_, dataset.get_training_df(), k, distance_functions)
    
    for i, table in enumerate(centroids_feature_map):
        logging.info(f"\nCluster {i}:\n{table}\n")

    # 8. Visualize the labeling from the clustering algorithm on the optimal feature set
    visualize_labeled_dataframe_transformation(dataset.get_training_df(), optimal_prototype_labels, k, optimal_indices, counts_table, config.STORE_IMAGES_DURING_EXECUTION, config.SHOW_IMAGES_DURING_EXECUTION)

if __name__ == "__main__":
    log_path = config.LOG_PATH + f"/{config.create_processed_data_path(False)}.log"
    utils.check_dir_file(log_path, True)
    logging.basicConfig(filename=log_path, level=config.LOG_LEVEL, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    main()
