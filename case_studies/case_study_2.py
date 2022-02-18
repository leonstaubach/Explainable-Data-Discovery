"""
Compares the optimal feature set to the same full feature set, but without the specific List Treatment
-> Lists are just treated as unique identifiers (categorical variable)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute()))

import logging
from numpy import sqrt
from multiprocessing import cpu_count
import src.utils as utils
import config
from src.kmodes.kprototypes import KPrototypes
from src.logic_part_2 import calculate_normalized_ch_index
from src.data_loader import CustomDataset

def main():
    logging.info('\n' + '-' * 15 + ' BEGINNING NEW PROCESS'  + '-' * 15)
    logging.info(utils.config_to_str())

    # Analyse relevant features from result_table
    utils.check_dir_file(config.PATH_OUTPUT_METADATA, True)
    meta_data = utils.load_meta_data_attribute(config.PATH_OUTPUT_METADATA)
    
    if "best_indices" not in meta_data:
        raise RuntimeError("Tried preloading best_indices, but didn't find the proper key in the metadata")
    if "feature_ranking" not in meta_data:
        raise RuntimeError("Tried preloading feature ranks, but didn't find the proper key in the metadata")
    if "k" not in meta_data:
        raise RuntimeError("Tried preloading k, but didn't find the proper key in the metadata")

    k = meta_data["k"]
    best_indices = meta_data["best_indices"]

    n_jobs = min(cpu_count(), config.K_PROTOTYPE_REPEAT_NUM)
    verbose = False

    dataset = CustomDataset(config.create_processed_data_path(True))
    
    # Update training df here based on the remaining feature ranks (keep names in order)
    dataset.update_training_df(sorted(meta_data.get("feature_ranking")))
    if len(best_indices["list_indices"]) == 0:
        indices_map = utils.create_indices_map(dataset.get_training_df().columns, dataset.get_training_df().attrs)
        # Hardcode index 1 for now
        best_indices["list_indices"] = [1]


    # Reload the dataset with the added option 'ignore_lists=True' to transform lists to categorical variables
    dataset_non_list = CustomDataset(config.create_processed_data_path(True), ignore_lists=True)
    
    # Bring over list indices to cat indices (since we treat lists as cat variables)
    indices_non_list = {k:v for k,v in best_indices.items()}


    indices_non_list["cat_indices"] = sorted(indices_non_list["cat_indices"] + indices_non_list["list_indices"])
    indices_non_list["list_indices"] = []
    dataset_non_list.update_training_df(sorted(meta_data.get("feature_ranking")))

    # INFO: Hardcode n_init value here, because i always want to have multiple runs for these

    logging.info(f"\nUsing k={k}")
    logging.info(f"\nStarting K-Prototypes with feature set: {best_indices} ..")        
    k_prototype_optimal = KPrototypes(n_clusters=k, init='Cao', n_init=12, verbose=verbose, n_jobs=n_jobs)
    _, costs_list, ch_index_optimal = k_prototype_optimal.fit_predict(dataset.get_training_df(), indices_map=best_indices)

    logging.info(f"\nStarting K-Prototypes with list-transformed feature set: {indices_non_list} ..") 
    k_prototype_optimal_non_list = KPrototypes(n_clusters=k, init='Cao', n_init=12, verbose=verbose, n_jobs=n_jobs)
    _, costs_non_list, ch_index_optimal_non_list = k_prototype_optimal_non_list.fit_predict(dataset_non_list.get_training_df(), indices_map=indices_non_list)

    nch_optimal = calculate_normalized_ch_index(dataset_non_list.get_training_df().values, ch_index_optimal, indices_non_list, k_prototype_optimal, False)
    nch_optimal_non_list = calculate_normalized_ch_index(dataset.get_training_df().values, ch_index_optimal_non_list, best_indices, k_prototype_optimal_non_list, False)
    logging.info(f"Hello: {nch_optimal} vs Transformed: {nch_optimal_non_list}")


    logging.info(f"\nNCH INDEX OPTIMAL: {nch_optimal} vs NCH INDEX OPTIMAL NON LIST: {nch_optimal_non_list}")
    logging.info(f"Improvement Absolute: {(nch_optimal-nch_optimal_non_list)}\nImprovement Relative: {utils.transform_to_percent((nch_optimal-nch_optimal_non_list)/nch_optimal_non_list)}%")

    nch_optimal = sqrt(nch_optimal)
    nch_optimal_non_list = sqrt(nch_optimal_non_list)
    logging.info(f"\nNow Showcasing Squarerooted Results")
    logging.info(f"\nNCH INDEX OPTIMAL: {nch_optimal} vs NCH INDEX OPTIMAL NON LIST: {nch_optimal_non_list}")
    logging.info(f"Improvement Absolute: {(nch_optimal-nch_optimal_non_list)}\nImprovement Relative: {utils.transform_to_percent((nch_optimal-nch_optimal_non_list)/nch_optimal_non_list)}%")

if __name__ == "__main__":
    log_path = config.LOG_PATH + f"/{config.create_processed_data_path(False)}_CASE_STUDY_2.log"
    utils.check_dir_file(log_path, True)
    logging.basicConfig(filename=log_path, level=config.LOG_LEVEL, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    main()
