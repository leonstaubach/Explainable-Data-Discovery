"""
Compares the optimal feature set to the same optimal feature set, but without the Time Treatment
-> Times are just treated as integers (numerical variable)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute()))

import logging
from multiprocessing import cpu_count
import src.utils as utils
import config
from src.data_loader import CustomDataset
from kmodes.kprototypes import KPrototypes

def main():
    logging.info('\n' + '-' * 15 + ' BEGINNING NEW PROCESS'  + '-' * 15)
    logging.info(utils.config_to_str())

    utils.check_dir_file(config.PATH_OUTPUT_METADATA, True)
    meta_data = utils.load_meta_data_attribute(config.PATH_OUTPUT_METADATA)
    

    if "feature_ranking" not in meta_data:
        raise RuntimeError("Tried preloading feature ranks, but didn't find the proper key in the metadata")
    if "best_indices" not in meta_data:
        raise RuntimeError("Tried preloading best_indices, but didn't find the proper key in the metadata")
    if "k" not in meta_data:
        raise RuntimeError("Tried preloading k, but didn't find the proper key in the metadata")
    best_indices = meta_data["best_indices"]

    if len(best_indices["time_indices"]) == 0:
        logging.info(f"\nCase Study 3 requires Time Indices, but none were found in the Metadata!")

    k = meta_data["k"]
    n_jobs = min(cpu_count(), config.K_PROTOTYPE_REPEAT_NUM)
    verbose = False

    dataset = CustomDataset(config.create_processed_data_path(True))

    # Update training df here based on the remaining feature ranks (keep names in order)
    dataset.update_training_df(sorted(meta_data["feature_ranking"]))
    logging.info(f"{dataset.get_training_df().columns}")

    # Reload the dataset with the added option 'ignore_times=True' to transform time variabiles to numerical variables
    dataset_non_time = CustomDataset(config.create_processed_data_path(True), ignore_times=True)
    dataset_non_time.update_training_df(sorted(meta_data["feature_ranking"]))

    best_indices_non_times = {k:v for k,v in best_indices.items()}
    # Bring over time indices to cat indices (since we treat lists as cat variables)
    best_indices_non_times["num_indices"] = best_indices_non_times["num_indices"] + best_indices_non_times["time_indices"] 
    best_indices_non_times["time_indices"] = []
  

    logging.info(f"\nStarting K-Prototypes with optimal feature set: {best_indices} ..")        
    # INFO: Hardcode n_init value here, because i always want to have multiple runs for these
    k_prototype_optimal = KPrototypes(n_clusters=k, init='Cao', n_init=5, verbose=verbose, n_jobs=n_jobs)
    _, _, ch_index_optimal = k_prototype_optimal.fit_predict(dataset.get_training_df(), indices_map=best_indices)


    """ 
    INFO: With time variables being transformed to numerical variables, the gamma value will be different from its original value.
        This is intended to properly compare numerical and non-numerical variables.
        Since the CH Index Calculation uses the ratio of intra and extra distances, different gamma values barely change the magnitude of the metric.
    """
 
    k_prototype_optimal_non_time = KPrototypes(n_clusters=k, init='Cao', n_init=5, verbose=verbose, n_jobs=n_jobs)
    _, _, ch_index_optimal_non_time = k_prototype_optimal_non_time.fit_predict(dataset_non_time.get_training_df(), indices_map=best_indices_non_times)
    

    logging.info(f"\nCH INDEX OPTIMAL: {ch_index_optimal} vs CH INDEX OPTIMAL NON TIMES: {ch_index_optimal_non_time}.\n")
    logging.info(f"\nImprovement Absolute: {(ch_index_optimal-ch_index_optimal_non_time)}\n Improvement Relative: {utils.transform_to_percent((ch_index_optimal-ch_index_optimal_non_time)/ch_index_optimal_non_time)}%")

if __name__ == "__main__":
    log_path = config.LOG_PATH + f"/{config.create_processed_data_path(False)}_CASE_STUDY_3.log"
    utils.check_dir_file(log_path, True)
    logging.basicConfig(filename=log_path, level=config.LOG_LEVEL, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    main()
