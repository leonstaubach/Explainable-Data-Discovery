"""
Compares the feature ranking process from the paper (old) vs mine (new)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute()))

import logging
from multiprocessing import cpu_count
import src.utils as utils
import config
config.PATH_OUTPUT_IMAGES = config.PATH_OUTPUT_IMAGES + "_CASE_STUDY_4"
utils.check_dir_file(config.PATH_OUTPUT_IMAGES, False)
from src.logic_part_2 import calculate_normalized_ch_index, execute_iterative_clustering
from src.data_loader import CustomDataset
from src.visualisation_utils import create_umap_plot
from kmodes.kprototypes import KPrototypes

def main():
    logging.info('\n' + '-' * 15 + ' BEGINNING NEW PROCESS'  + '-' * 15)
    logging.info(utils.config_to_str())

    # Analyse relevant features from result_table
    utils.check_dir_file(config.PATH_OUTPUT_METADATA, True)
    meta_data = utils.load_meta_data_attribute(config.PATH_OUTPUT_METADATA)
    
    # 1. Load Data
    dataset = CustomDataset(config.create_processed_data_path(True))
    df = dataset.get_training_df()   
    
    
    if "feature_ranking" not in meta_data:
        raise RuntimeError("Tried preloading feature ranks, but didn't find the proper key in the metadata")
    if "old_feature_ranking" not in meta_data:
        raise RuntimeError("Tried preloading old_feature_ranking, but didn't find the proper key in the metadata")
    if "best_indices" not in meta_data:
        raise RuntimeError("Tried preloading best_indices, but didn't find the proper key in the metadata")
    if "k" not in meta_data:
        raise RuntimeError("Tried preloading k, but didn't find the proper key in the metadata")

    """ Update training df based on feature ranks. Since both feature_ranking and old_feature_ranking have the same initial
        noisy features removed, the dataset can be reused for both processes.
    """
    dataset.update_training_df(sorted(meta_data["feature_ranking"]))
    
    best_indices = meta_data["best_indices"]
    
    if 'best_old_indices' not in meta_data:
        optimal_indices_old = execute_iterative_clustering(dataset.get_training_df(), meta_data, True)
        meta_data["best_old_indices"] = optimal_indices_old
        utils.update_meta_data_attribute(f"{config.PATH_OUTPUT_METADATA}", "best_old_indices", optimal_indices_old)

    else:
        optimal_indices_old = meta_data['best_old_indices']

    n_jobs = min(cpu_count(), config.K_PROTOTYPE_REPEAT_NUM)
    verbose = False
    k = meta_data["k"]

    # INFO: Hardcode n_init value here, because i always want to have multiple runs for these
    logging.info(f"\nStarting K-Prototypes with old ranked feature set: {optimal_indices_old} ..")   
    k_prototype_old = KPrototypes(n_clusters=k, init='Cao', n_init=5, verbose=verbose, n_jobs=n_jobs)
    labels_old, _, ch_index_old = k_prototype_old.fit_predict(dataset.get_training_df(), indices_map=optimal_indices_old)


    logging.info(f"\nStarting K-Prototypes with new ranked feature set: {best_indices} ..")        
    k_prototype_optimal = KPrototypes(n_clusters=k, init='Cao', n_init=5, verbose=verbose, n_jobs=n_jobs)
    best_prototype_labels, _, ch_index_optimal = k_prototype_optimal.fit_predict(dataset.get_training_df(), indices_map=best_indices)



    X = dataset.get_training_df().values
    nch_new = calculate_normalized_ch_index(X, ch_index_optimal, optimal_indices_old, k_prototype_optimal)
    nch_old = calculate_normalized_ch_index(X, ch_index_old, best_indices, k_prototype_old)
    logging.info(f"\nCH INDEX NEW: {nch_new} vs CH INDEX OLD: {nch_old}.")
    logging.info(f"\nImprovement Absolute: {(nch_new-nch_old)}\nImprovement Relative: {utils.transform_to_percent((nch_new-nch_old)/nch_old)}%")

    if config.STORE_IMAGES_DURING_EXECUTION or config.SHOW_IMAGES_DURING_EXECUTION:
        utils.check_dir_file(config.PATH_OUTPUT_IMAGES, False)

        path = f"{config.PATH_OUTPUT_IMAGES}/3_0_OLD_FEATURE_k={k}.png"

        create_umap_plot(dataset.get_training_df(), labels_old, k, optimal_indices_old, config.STORE_IMAGES_DURING_EXECUTION, config.SHOW_IMAGES_DURING_EXECUTION, path)    
        path = f"{config.PATH_OUTPUT_IMAGES}/3_1_NEW_FEATURE_k={k}.png"
        create_umap_plot(dataset.get_training_df(), best_prototype_labels, k, best_indices, config.STORE_IMAGES_DURING_EXECUTION, config.SHOW_IMAGES_DURING_EXECUTION, path)


if __name__ == "__main__":
    log_path = config.LOG_PATH + f"/{config.create_processed_data_path(False)}_CASE_STUDY_4.log"
    utils.check_dir_file(log_path, True)
    logging.basicConfig(filename=log_path, level=config.LOG_LEVEL, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    main()
