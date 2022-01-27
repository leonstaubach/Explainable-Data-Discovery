"""
Compares the full feature set vs the optimal feature set
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute()))

import logging
from multiprocessing import cpu_count
import src.utils as utils
import config
config.PATH_OUTPUT_IMAGES = config.PATH_OUTPUT_IMAGES + "_CASE_STUDY_1"
utils.check_dir_file(config.PATH_OUTPUT_IMAGES, False)
from src.logic_part_2 import calculate_normalized_ch_index
from src.data_loader import CustomDataset
from src.visualisation_utils import create_umap_plot
from src.kmodes.kprototypes import KPrototypes

def main():
    logging.info('\n' + '-' * 15 + ' BEGINNING NEW PROCESS'  + '-' * 15)
    logging.info(utils.config_to_str())

    utils.check_dir_file(config.PATH_OUTPUT_METADATA, True)
    meta_data = utils.load_meta_data_attribute(config.PATH_OUTPUT_METADATA)
    
    dataset = CustomDataset(config.create_processed_data_path(True))

    if "feature_ranking" not in meta_data:
        raise RuntimeError("Tried preloading feature_ranking, but didn't find the proper key in the metadata")

    # Update training df here based on the remaining feature ranks (keep names in order)
    dataset.update_training_df(sorted(meta_data.get("feature_ranking")))


    if "best_indices" not in meta_data:
        raise RuntimeError("Tried preloading best_indices, but didn't find the proper key in the metadata")
    indices_map = utils.create_indices_map(dataset.get_training_df().columns, dataset.get_training_df().attrs)

    best_indices = meta_data["best_indices"]


    n_jobs = min(cpu_count(), config.K_PROTOTYPE_REPEAT_NUM)
    verbose = False
    k = meta_data["k"]

    # INFO: Hardcode n_init value here, because i always want to have multiple runs for these
    k_prototype_full = KPrototypes(n_clusters=k, init='Cao', n_init=5, verbose=verbose, n_jobs=n_jobs)
    best_prototype_labels, _, ch_index_full = k_prototype_full.fit_predict(dataset.get_training_df(), indices_map=indices_map)

    logging.info(f"\nStarting K-Prototypes with optimal feature set: {best_indices} ..")   
    # Grab initial centroids from previous run for comparability, delete unused features     
    initial_centroids = utils.update_centroids(k_prototype_full.newly_initialized_centroids, indices_map, best_indices)
    
    k_prototype_optimal = KPrototypes(n_clusters=k, init='Cao', n_init=5, verbose=verbose, n_jobs=n_jobs, initial_centroids=initial_centroids)
    best_prototype_labels_optimal, _, ch_index_optimal = k_prototype_optimal.fit_predict(dataset.get_training_df(), indices_map=best_indices)

    X = dataset.get_training_df().values
    full_nch = calculate_normalized_ch_index(X, ch_index_full, best_indices, k_prototype_full)
    optimal_nch = calculate_normalized_ch_index(X, ch_index_optimal, indices_map, k_prototype_optimal)
    logging.info(f"\nFull Dataset NCH: {full_nch} vs Optimal NCH: {optimal_nch}")

    if config.STORE_IMAGES_DURING_EXECUTION or config.SHOW_IMAGES_DURING_EXECUTION:
        utils.check_dir_file(config.PATH_OUTPUT_IMAGES, False)

        path = f"{config.PATH_OUTPUT_IMAGES}/3_0_FULL_FEATURE_k={k}.png"

        create_umap_plot(dataset.get_training_df(), best_prototype_labels, k, indices_map, config.STORE_IMAGES_DURING_EXECUTION, config.SHOW_IMAGES_DURING_EXECUTION, path)    
        path = f"{config.PATH_OUTPUT_IMAGES}/3_1_OPTIMAL_FEATURE_k={k}.png"

        create_umap_plot(dataset.get_training_df(), best_prototype_labels_optimal, k, best_indices, config.STORE_IMAGES_DURING_EXECUTION, config.SHOW_IMAGES_DURING_EXECUTION, path)


if __name__ == "__main__":
    log_path = config.LOG_PATH + f"/{config.create_processed_data_path(False)}_CASE_STUDY_1.log"
    utils.check_dir_file(log_path, True)
    logging.basicConfig(filename=log_path, level=config.LOG_LEVEL, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    main()
