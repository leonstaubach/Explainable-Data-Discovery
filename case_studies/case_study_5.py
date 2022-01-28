"""
Compares different configs for UMAP. More of a on-demand testing script.


-> TL;DR:
        Hamming Distance is sufficient
        n_neighbors of 35-50 is sufficient (UMap fails to create the embedding with a zsh: bus error for lower n_neighbors)
        min_dist is less important, can be set to default of 0.1
        metric of "random" is quicker, but results in "low quality", so keep it on default
        
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute()))

import logging
from multiprocessing import cpu_count
import src.utils as utils
import config
config.PATH_OUTPUT_IMAGES = config.PATH_OUTPUT_IMAGES + "_CASE_STUDY_5"
utils.check_dir_file(config.PATH_OUTPUT_IMAGES, False)
from src.logic_part_2 import calculate_normalized_ch_index
from src.data_loader import CustomDataset
from src.visualisation_utils import create_umap_plot
from src.kmodes.kprototypes import KPrototypes

def main():
    logging.info('\n' + '-' * 15 + ' BEGINNING NEW PROCESS'  + '-' * 15)
    logging.info(utils.config_to_str())

    # Analyse relevant features from result_table
    utils.check_dir_file(config.PATH_OUTPUT_METADATA, True)
    meta_data = utils.load_meta_data_attribute(config.PATH_OUTPUT_METADATA)
    
    # 1. Load Data
    dataset = CustomDataset(config.create_processed_data_path(True))

    if "feature_ranking" not in meta_data:
        raise RuntimeError("Tried preloading feature ranks, but didn't find the proper key in the metadata")
    if "best_indices" not in meta_data:
        raise RuntimeError("Tried preloading best_indices, but didn't find the proper key in the metadata")
    if "k" not in meta_data:
        raise RuntimeError("Tried preloading k, but didn't find the proper key in the metadata")
    dataset.update_training_df(sorted(meta_data["feature_ranking"]))
    
    indices_map = utils.create_indices_map(dataset.get_training_df().columns, dataset.get_training_df().attrs)
    best_indices = meta_data["best_indices"]

    n_jobs = min(cpu_count(), config.K_PROTOTYPE_REPEAT_NUM)
    verbose = False
    k = meta_data["k"]

    # INFO: Hardcode n_init value here, because i always want to have multiple runs for these
    logging.info(f"\nStarting K-Prototypes with full feature set: {indices_map} ..")     
    k_prototype_full = KPrototypes(n_clusters=k, init='Cao', n_init=1, verbose=verbose, n_jobs=n_jobs)
    best_prototype_labels, _, ch_index_full = k_prototype_full.fit_predict(dataset.get_training_df(), indices_map=indices_map)

    logging.info(f"\nStarting K-Prototypes with optimal feature set: {best_indices} ..")        
    k_prototype_optimal = KPrototypes(n_clusters=k, init='Cao', n_init=1, verbose=verbose, n_jobs=n_jobs)
    best_prototype_labels_optimal, _, ch_index_optimal = k_prototype_optimal.fit_predict(dataset.get_training_df(), indices_map=best_indices)

    X = dataset.get_training_df().values
    full_nch = calculate_normalized_ch_index(X, ch_index_full, best_indices, k_prototype_full)
    optimal_nch = calculate_normalized_ch_index(X, ch_index_optimal, indices_map, k_prototype_optimal)
    logging.info(f"\nFull Dataset NCH: {full_nch} vs Optimal NCH: {optimal_nch}")


    checkable_distance_functions = ["hamming", "jaccard", "dice", "kulsinski", "ll_dirichlet", "hellinger", "rogerstanimoto", "sokalmichener", "sokalsneath", "yule"]
    n_neighbors = [10, 15, 20, 25, 30, 35, 45, 50]
    min_dists = [0.0, 0.1, 0.2, 0.5, 0.7, 0.99]
    n_neighbors.reverse()
    min_dists.reverse()
    
    if config.STORE_IMAGES_DURING_EXECUTION or config.SHOW_IMAGES_DURING_EXECUTION:
        for min_dist in min_dists:   
            for dist_fct in checkable_distance_functions:
                for n_neighbor in n_neighbors:
                    utils.check_dir_file(config.PATH_OUTPUT_IMAGES, False)

                    try:
                        path = f"{config.PATH_OUTPUT_IMAGES}/3_0_FULL_FEATURE_k={k}_{dist_fct}_n_neighb_{n_neighbor}_mindist_{min_dist}.png"  
                        create_umap_plot(dataset.get_training_df(), best_prototype_labels, k, indices_map, config.STORE_IMAGES_DURING_EXECUTION, config.SHOW_IMAGES_DURING_EXECUTION, path, dist_fct, n_neighbor, min_dist)    
                    except Exception:
                        logging.info(f"\nUMap Failed for {min_dist} on the full feature set")
                    
                    try:
                        path = f"{config.PATH_OUTPUT_IMAGES}/3_1_OPTIMAL_FEATURE_k={k}_{dist_fct}_n_neighb_{n_neighbor}_mindist_{min_dist}.png"

                        create_umap_plot(dataset.get_training_df(), best_prototype_labels_optimal, k, best_indices, config.STORE_IMAGES_DURING_EXECUTION, config.SHOW_IMAGES_DURING_EXECUTION, path, dist_fct, n_neighbor)
                    except Exception:
                        logging.info(f"\nUMap Failed for {min_dist} on the optimal feature set")

if __name__ == "__main__":
    log_path = config.LOG_PATH + f"/{config.create_processed_data_path(False)}_CASE_STUDY_5.log"
    utils.check_dir_file(log_path, True)
    logging.basicConfig(filename=log_path, level=config.LOG_LEVEL, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    main()
