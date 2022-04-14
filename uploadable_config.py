from logging import INFO, DEBUG, FATAL, WARN
from datetime import datetime

############################################################################################################
# MUST-SET VARIABLES, for loading the data correctly.
# Parqet-File required in the name-format of {API_KEY}_{TARGET_ACTIVITY_TYPE}_{COMPRESSION_ALGORITHM}.parqet"
#
# The current apiKey
API_KEY = "NameOfTheApiKey"
# The current activity type
TARGET_ACTIVITY_TYPE = "NameOfTheActivityType"
# Optional (if multiple version of the data exists, e.g. {'Full', 'Medium', 'Test', ...})
TEST_MODE_CATEGORY = "DEFAULT"
# Compression algorithm used for the data (for logging and path resolving only)
COMPRESSION_ALGORITHM="gzip"
# Relative path of the parqet file (for example the local folder 'data'
BASE_PROCESSED_DATA_FOLDER = "data"
############################################################################################################

############################################################################################################
# SHOULD-SET VARIABLES, Depending on the use-case
#
# Whether to stop the execution of the process to show the plots
SHOW_IMAGES_DURING_EXECUTION=False
# Whether to store the plots to disk
STORE_IMAGES_DURING_EXECUTION=True
# Format of the time variable (should be a format that can be directly transformed to an integer,
# should be updated in the future, since it is a very manual approach)
TIME_FORMAT = "%H"
# Whether to preload features from the metadata file or not.
TRY_PRELOAD_FEATURES = True
# Whether to transform feature names to integers (for anonymization reasons)
ANONIMIZE_FEATURE_NAMES=True
# Max number of clusters to check for the elbow method
MAX_NUM_CLUSTERS_ELBOW = 10
# How often a k-Prototype run should be repeated (to find optimal solutions for partially random inializations)
K_PROTOTYPE_REPEAT_NUM = 5
# Whether to update the gamma value (weight of non-numerical variables) each iteration or not.
# False is recommended per default for comparability between iterations of clustering.
# True can be used for experimental reasons.
UPDATE_GAMMA_EACH_ITERATION = False
# Which formula to use for the feature ranking. Options: {"formula_updated", formula+1", "formula_normalized"}
# , see src/metrics.py
FEATURE_IMPORTANCE_METRIC = "formula_updated"
# Whether to remove outliers (based on the Z-Score metric with a threshold of 3)
REMOVE_OUTLIERS = True
#############################################################################################################

#############################################################################################################
# CAN-SET VARIABLES
#
# Maximum number of bins per numerical variabile (to calculate Entropies etc.)
MAX_BIN_NUMBER = 1000
# Percentage of number of total values of a feature used for binning
BIN_RATIO = 0.3
# Format of current time to create folders for each execution
TIME = datetime.today().strftime("%Y-%m-%d_%H::%M::%S")
# The relative path to store logs in
LOG_PATH = "logs"
# The log level
LOG_LEVEL = INFO
# Path to write the images to 
PATH_OUTPUT_IMAGES = f"output/{API_KEY}/{TARGET_ACTIVITY_TYPE}/{TEST_MODE_CATEGORY.lower()}/images/{TIME}"
# Path to write the metadata to
PATH_OUTPUT_METADATA = f"output/{API_KEY}/{TARGET_ACTIVITY_TYPE}/{TEST_MODE_CATEGORY.lower()}/metadata.json"
# Max number of characters per label for the plots
MAX_CHARACTERS_TO_DISPLAY=20
# Max columns per total plot
MAX_COLS_TO_DISPLAY=3
# Ratio of the display
DISPLAY_SIZE=(20,9)
# How many DPI's should be used for the image
DPI_TO_STORE=350
# UMaps n_neighbors parameter
UMAP_PARAM_N_NEIGHBORS=15
# UMaps min_Dist parameter
UMAP_PARAM_MIN_DIST=0.1
# UMaps metric parameter
UMAP_PARAM_METRIC="hamming"
################################################################################################################


def create_raw_data_path(add_compression=True):
    # A flag depending on some files. Sometimes we want the compression type in the file.
    result = f"{API_KEY}_{TARGET_ACTIVITY_TYPE}"
    if add_compression:
        result += f"_{COMPRESSION_ALGORITHM}"

    return result


def create_processed_data_path(add_compression=True, include_test_mode=True):
    result = create_raw_data_path(add_compression) 
    if include_test_mode:
        if TEST_MODE_CATEGORY == "Small":
            result += "_Small"
        elif TEST_MODE_CATEGORY == "Medium":
            result += "_Medium"
        elif TEST_MODE_CATEGORY == "Full":
            result += "_Full"

    return result

# Describes columns of the current data set that are not useful for training, but might be useful for evaluation
IGNORABLE_FEATURES_FOR_EVAL = []

# Any additional columns names that should be dropped before data processing
IMMEDIATLY_DROPPABLE_COLUMNS = []

# Any additional columns names that should be dropped after data processing
LATER_DROPPABLE_COLUMNS = []

# Used for a clean way of printing all relevant configs to understand a evaluation step
def return_current_config_dict():
    return {
        "DATA_SETUP_PARAMETERS": {
            "API_KEY": API_KEY,     
            "TARGET_ACTIVITY_TYPE": TARGET_ACTIVITY_TYPE,
            "COMPRESSION_ALGORITHM": COMPRESSION_ALGORITHM,
            "TEST_MODE_CATEGORY": TEST_MODE_CATEGORY,
            "MAX_BIN_NUMBER": MAX_BIN_NUMBER,
            "BIN_RATIO": BIN_RATIO,
            "TIME_FORMAT": TIME_FORMAT,
        },
        "STEP_1_PARAMETERS": {
            "TRY_PRELOAD_FEATURES": TRY_PRELOAD_FEATURES
        },
        "STEP_2_PARAMETERS": {
            "K_PROTOTYPE_REPEAT_NUM" : K_PROTOTYPE_REPEAT_NUM,
            "MAX_NUM_CLUSTERS_ELBOW" : MAX_NUM_CLUSTERS_ELBOW
        },
        "POST_EVALUATION": {
            "FEATURE_IMPORTANCE_METRIC" : FEATURE_IMPORTANCE_METRIC
        }
    }
