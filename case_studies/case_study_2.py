"""
Compares the optimal feature set to the same optimal feature set, but without the specific List Treatment
-> Lists are just treated as unique identifiers (categorical variable)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute()))

import logging
from multiprocessing import cpu_count
import src.utils as utils
import config
from src.logic_part_2 import calculate_normalized_ch_index
from src.data_loader import CustomDataset
from src.visualisation_utils import create_umap_plot, create_cluster_table, visualize_labeled_dataframe_transformation
from src.metrics import create_explicit_feature_importance
from kmodes.kprototypes import KPrototypes
from kmodes.util.dissim import matching_dissim, euclidean_dissim, matching_dissim_lists, time_dissim
def main():
    logging.info('\n' + '-' * 15 + ' BEGINNING NEW PROCESS'  + '-' * 15)
    logging.info(utils.config_to_str())

    # Analyse relevant features from result_table
    utils.check_dir_file(config.PATH_OUTPUT_METADATA, True)
    meta_data = utils.load_meta_data_attribute(config.PATH_OUTPUT_METADATA)
    

    if "feature_ranking" not in meta_data:
        raise RuntimeError("Tried preloading feature ranks, but didn't find the proper key in the metadata")
    if "best_indices" not in meta_data:
        raise RuntimeError("Tried preloading best_indices, but didn't find the proper key in the metadata")
    if "k" not in meta_data:
        raise RuntimeError("Tried preloading k, but didn't find the proper key in the metadata")
    best_indices = meta_data["best_indices"]

    if len(best_indices["list_indices"]) == 0:
        raise RuntimeError(f"Case Study 2 requires List Indices, but none were found in the Metadata!")

    k = meta_data["k"]

    n_jobs = min(cpu_count(), config.K_PROTOTYPE_REPEAT_NUM)
    verbose = False

    dataset = CustomDataset(config.create_processed_data_path(True))
    
    # Update training df here based on the remaining feature ranks (keep names in order)
    dataset.update_training_df(sorted(meta_data.get("feature_ranking")))

    # Reload the dataset with the added option 'ignore_lists=True' to transform lists to categorical variables
    dataset_non_list = CustomDataset(config.create_processed_data_path(True), ignore_lists=True)
    
    # Bring over list indices to cat indices (since we treat lists as cat variables)
    best_indices_non_list = {k:v for k,v in best_indices.items()}
    best_indices_non_list["cat_indices"] = best_indices_non_list["cat_indices"] + best_indices_non_list["list_indices"] 
    best_indices_non_list["list_indices"] = []
    dataset_non_list.update_training_df(sorted(meta_data.get("feature_ranking")))

    # INFO: Hardcode n_init value here, because i always want to have multiple runs for these
    logging.info(f"\nStarting K-Prototypes with optimal feature set: {best_indices} ..")        
    k_prototype_optimal = KPrototypes(n_clusters=k, init='Cao', n_init=5, verbose=verbose, n_jobs=n_jobs)
    _, _, ch_index_optimal = k_prototype_optimal.fit_predict(dataset.get_training_df(), indices_map=best_indices)

    logging.info(f"\nStarting K-Prototypes with optimal list-transformed feature set: {best_indices} ..") 
    k_prototype_optimal_non_list = KPrototypes(n_clusters=k, init='Cao', n_init=5, verbose=verbose, n_jobs=n_jobs)
    _, _, ch_index_optimal_non_list = k_prototype_optimal_non_list.fit_predict(dataset_non_list.get_training_df(), indices_map=best_indices_non_list)

    logging.info(f"\nCH INDEX OPTIMAL: {ch_index_optimal} vs CH INDEX OPTIMAL NON LIST: {ch_index_optimal_non_list}.\n")
    logging.info(f"\nImprovement of using custom list - Absolute: {(ch_index_optimal-ch_index_optimal_non_list)}\nRelative: {utils.transform_to_percent((ch_index_optimal-ch_index_optimal_non_list)/ch_index_optimal_non_list)}%")

if __name__ == "__main__":
    log_path = config.LOG_PATH + f"/{config.create_processed_data_path(False)}_CASE_STUDY_2.log"
    utils.check_dir_file(log_path, True)
    logging.basicConfig(filename=log_path, level=config.LOG_LEVEL, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    main()
