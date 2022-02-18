"""
Compares the optimal feature set to the same optimal feature set, but without the Time Treatment
-> Times are just treated as integers (numerical variable)
"""

import sys
from pathlib import Path

from numpy import sqrt
sys.path.insert(0, str(Path('.').absolute()))

import logging
from multiprocessing import cpu_count
import src.utils as utils
import config
from src.data_loader import CustomDataset
from src.kmodes.kprototypes import KPrototypes
from src.logic_part_2 import calculate_normalized_ch_index
def main():
    logging.info('\n' + '-' * 15 + ' BEGINNING NEW PROCESS'  + '-' * 15)
    logging.info(utils.config_to_str())

    utils.check_dir_file(config.PATH_OUTPUT_METADATA, True)
    meta_data = utils.load_meta_data_attribute(config.PATH_OUTPUT_METADATA)
    

    if "feature_ranking" not in meta_data:
        raise RuntimeError("Tried preloading feature ranks, but didn't find the proper key in the metadata")

    n_jobs = min(cpu_count(), config.K_PROTOTYPE_REPEAT_NUM)
    verbose = False

    dataset = CustomDataset(config.create_processed_data_path(True))

    # Update training df here based on the remaining feature ranks (keep names in order)
    dataset.update_training_df(sorted(meta_data["feature_ranking"]))

    logging.info(f"{dataset.get_training_df().columns}")
    if "best_indices" not in meta_data:
        raise RuntimeError("Tried preloading best_indices, but didn't find the proper key in the metadata")
    if "k" not in meta_data:
        raise RuntimeError("Tried preloading k")
    k = meta_data["k"]
    best_indices = meta_data["best_indices"]
    if len(best_indices["time_indices"]) == 0:
        indices_map = utils.create_indices_map(dataset.get_training_df().columns, dataset.get_training_df().attrs)
        best_indices["time_indices"] = indices_map["time_indices"]
        best_indices["time_max_values"] = indices_map["time_max_values"]
    # Reload the dataset with the added option 'ignore_times=True' to transform time variabiles to numerical variables
    dataset_non_time = CustomDataset(config.create_processed_data_path(True), ignore_times=True)
    dataset_non_time.update_training_df(sorted(meta_data["feature_ranking"]))
    print(dataset_non_time.get_training_df().head())
    indices_non_times = {k:v for k,v in best_indices.items()}
    # Bring over time indices to cat indices (since we treat lists as cat variables)
    indices_non_times["num_indices"] = sorted(indices_non_times["num_indices"] + indices_non_times["time_indices"])
    indices_non_times["time_indices"] = []
    indices_non_times["time_max_values"] = []
  
    logging.info(f"\nUsing k={k}")
    logging.info(f"\nStarting K-Prototypes with feature set: {best_indices} ..")        
    # INFO: Hardcode n_init value here, because i always want to have multiple runs for these
    k_prototype_optimal = KPrototypes(n_clusters=k, init='Cao', n_init=12, verbose=verbose, n_jobs=n_jobs)
    _, costs_time, ch_index_optimal = k_prototype_optimal.fit_predict(dataset.get_training_df(), indices_map=best_indices)

    """ 
    INFO: With time variables being transformed to numerical variables, the gamma value will be different from its original value.
        This is intended to properly compare numerical and non-numerical variables.
        Since the CH Index Calculation uses the ratio of intra and extra distances, different gamma values barely change the magnitude of the metric.
    """
    logging.info(f"\nStarting K-Prototypes with transformed feature set: {indices_non_times} ..") 
    k_prototype_optimal_non_time = KPrototypes(n_clusters=k, init='Cao', n_init=12, verbose=verbose, n_jobs=n_jobs)#, gamma=k_prototype_optimal.gamma)
    _, costs_non_time, ch_index_optimal_non_time = k_prototype_optimal_non_time.fit_predict(dataset_non_time.get_training_df(), indices_map=indices_non_times)
    

    nch_optimal = calculate_normalized_ch_index(dataset_non_time.get_training_df().values, ch_index_optimal, indices_non_times, k_prototype_optimal, False)
    nch_optimal_non_time = calculate_normalized_ch_index(dataset.get_training_df().values, ch_index_optimal_non_time, best_indices, k_prototype_optimal_non_time, False)
    logging.info(f"Hello: {nch_optimal} vs Transformed: {nch_optimal_non_time}")


    logging.info(f"\nNCH INDEX OPTIMAL: {nch_optimal} vs NCH INDEX OPTIMAL NON TIMES: {nch_optimal_non_time}")
    logging.info(f"Improvement Absolute: {(nch_optimal-nch_optimal_non_time)}\nImprovement Relative: {utils.transform_to_percent((nch_optimal-nch_optimal_non_time)/nch_optimal_non_time)}%")

    nch_optimal = sqrt(nch_optimal)
    nch_optimal_non_time = sqrt(nch_optimal_non_time)
    logging.info(f"\nNow Showcasing Squarerooted Results")
    logging.info(f"\nNCH INDEX OPTIMAL: {nch_optimal} vs NCH INDEX OPTIMAL NON TIMES: {nch_optimal_non_time}")
    logging.info(f"Improvement Absolute: {(nch_optimal-nch_optimal_non_time)}\nImprovement Relative: {utils.transform_to_percent((nch_optimal-nch_optimal_non_time)/nch_optimal_non_time)}%")



if __name__ == "__main__":
    log_path = config.LOG_PATH + f"/{config.create_processed_data_path(False)}_CASE_STUDY_3.log"
    utils.check_dir_file(log_path, True)
    logging.basicConfig(filename=log_path, level=config.LOG_LEVEL, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    main()
