import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute()))

import datetime
import seaborn as sns
import numpy as np
import json
import os
from numbers import Number
from copy import deepcopy

from config import return_current_config_dict

# Just a quick definition for a custom exception..
class WouldTryRandomInitialization(Exception):
    """Base class for other exceptions"""
    pass

# Mapper for the ordering of indices (not that Python3 Dict's keep the keys in order)
MAPPER = {
        "num_indices": 0,
        "cat_indices": 1,
        "list_indices": 2,
        "time_indices": 3
}

def _format_arg_str(table_header: str, args: dict, exclude_lst: list = [], max_len=20) -> str:
    """ Formats a given config into a nice view. Copied from this repository: https://github.com/THUwangcy/ReChorus/blob/master/src/utils/utils.py#L59-L80
    
    
    :param table_header:    Header name of the table  
    :param args:            Arguments in the table
    :param exclude_lst:     Arguments that shouldn't be listed
    :param max_len:         Maximum characters to display per value and key

    :returns a nice string view of the given config
    """
    keys = [k for k in args.keys() if k not in exclude_lst]
    values = [args[k] for k in keys]
    table_header = " ".join([word.capitalize() for word in table_header.split("_")])
    key_title, value_title = table_header.replace("_", " "), 'Values'
    key_max_len = max(map(lambda x: len(str(x)), keys))
    value_max_len = min(max(map(lambda x: len(str(x)), values)), max_len)
    key_max_len, value_max_len = max([len(key_title), key_max_len]), max([len(value_title), value_max_len])
    horizon_len = key_max_len + value_max_len + 5
    res_str = os.linesep + '=' * horizon_len + os.linesep
    res_str += ' ' + key_title + ' ' * (key_max_len - len(key_title)) + ' | ' \
               + value_title + ' ' * (value_max_len - len(value_title)) + ' ' + os.linesep + '=' * horizon_len + os.linesep
    for key in sorted(keys):
        value = args[key]
        if value is not None:
            key, value = str(key).lower(), str(value).replace('\t', '\\t')
            value = value[:max_len-3] + '...' if len(value) > max_len else value
            res_str += ' ' + key + ' ' * (key_max_len - len(key)) + ' | ' \
                       + value + ' ' * (value_max_len - len(value)) + os.linesep
    res_str += '=' * horizon_len
    return res_str

def _format_data_config(args: dict):
    """ A helper to create a nicer view of the given config.

    :param args:    A dictionary of different tables to visualize.

    :returns a nicely formatted view of the given tables.
    """
    tables = []
    for key, data in args.items():
        if len(data) > 0:
            tables.append(_format_arg_str(key, data))

    return "\n".join(tables) + "\n"

def config_to_str():
    """ Wrapper to call the formatter of the current config.
    
    :returns the current config in string format.
    """
    return _format_data_config(return_current_config_dict())



def check_dir_file(path: str, is_file: bool) -> None:
    """
    Checks whether the nested dir path exists. If not, it will create the folder and all subfolders.

    :param path:        Relative patht to the file or folder
    :param is_file:     Whether the path is a file or a path
    """
    if is_file:
        dir_path = os.path.dirname(path)
    else:
        dir_path = path
        
    if not os.path.exists(dir_path):
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def load_meta_data_attribute(path: str):
    """ Load the mata data from the given path.
    
    :param path:    The path to the metadata file

    :returns the dictionary representing the metadata.
    """
    if not os.path.exists(path):
        return {}
        
    with open(path) as json_file:
        return json.load(json_file)
        
def update_meta_data_attribute(path: str, key:str, value: object):
    """ Update the meta data with the given key-value pair. Writes the updated json to disk.
    
    :param path:    Path to the metadata
    :param key:     Writable key
    :param value:   Writable value
    
    """
    if not os.path.exists(path):
        data = {}
    else:
        with open(path) as json_file:
            data = json.load(json_file)

    data[key] = value

    with open(path, 'w') as json_file:
        json.dump(data, json_file, indent=2)

def create_indices_map(columns, attrs):
    """ Creates the indices map based on the given columns and attributes of the data frame

    :param columns: Feature names of the dataset
    :param attrs:   Attributes dictionary of the dataset
    
    :returns a summarized indices map
    """
    numerical_indices = []
    categorical_indices = []
    list_indices = []
    time_indices = []
    time_max_values = []
    for i, column in enumerate(columns):
        type = attrs[column]["datatype"]
        if type == list:
            list_indices.append(i)
        elif type == Number:
            numerical_indices.append(i)
        elif type == datetime.date:
            time_indices.append(i)
            time_max_values.append(attrs[column]["max_time_value"])
        elif type == str:
            categorical_indices.append(i)
        else:
            raise RuntimeError(f"Unknown datatype for dataframe: {type}")

    return build_indices_map(numerical_indices, categorical_indices, list_indices, time_indices, time_max_values)

def get_time():
    """ Returns the current time in the given format """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def flatten_np_frozenset(data: np.array):
    """ Helper to create a flat numpy array from an array of frozen sets.
    
    :param data:    1D Numpy array of frozensets 
    
    :returns flattened list of each entry in the whole data
    """
    v = []
    for frset in data:
        for value in frset:
            v.append(value)

    return np.array(v, dtype=np.int32)

def first_peak(data):
    """ Returns the index of the first peak in a series of values. A peak is defined as an increase in value compared to the previous value.

    :param data:    List of values

    :returns index of the first increase within the data. If none found, returns 0.
    """
    prev = -1
    for i, value in enumerate(data):
        if prev != -1:
            if value > prev:
                return i
        prev = value
    # data is monotonically decreasing, return the highest value, which is index 0
    return i if i < len(data)-1 else 0 
  
def get_type_through_index(indices_map, current_index):
    """ Returns the internal index for a given indic within the given indices map.

    :param indices_map:     Dict containing the given indices for each data type.
    :param current_index:   Index that should be found in the indices_map

    :returns the index associated to the category of the current_index in the indices_map
    """
    for name, indices in indices_map.items():
        if name != "time_max_values":
            if current_index in indices:
                return MAPPER[name]

    raise RuntimeError("Something weird happened here, should've found the index...")

def resolve_index(data_type_index):
    """ Resolve the name of a given index for the different datatypes.

    :param data_type_index:   Integer representing the desired index of the datatype.

    :returns the name of the datatype.
    
    """
    for key, value in MAPPER.items():
        if data_type_index == value:
            return key

    raise RuntimeError("Something weird happened here, should've found the index...")


def build_indices_map(num, cat, lists, times, time_max_values):
    """ Build the indices map from the given list of indices
    
    :param num, cat, lists, times:  Arrays of indices for each datatype
    :param time_max_values:         List of max time values for all time variables

    :returns summarized indices map
    """
    return {
        "num_indices":num,
        "cat_indices":cat,
        "list_indices":lists,
        "time_indices":times,
        "time_max_values": time_max_values
    }


def add_feature_to_indices_map(full_indices_map, new_feature, columns, indices_map, old_centroids, original_centroids):
    """ Adds a feature back to the indices map. Adjusts the given centroids accordingly.
    
    :param full_indices_map:        The original full map of indices
    :param new_feature:             The name of the feature that should be added
    :param columns:                 The full list of feature names
    :param indices_map:             The current indices map (which the feature will be added to)
    :param old_centroids:                 The current centroids
    :param original_centroids:            The original full list of centroids

    :returns the updated indices_map and the new centroids
    """
    indices_map_next = deepcopy(indices_map)
    target_index = columns.index(new_feature)
    new_centroids = deepcopy(old_centroids)

    i=0
    for key, value in full_indices_map.items():
        if i > 3:
            raise RuntimeError(f"Didn't expect to not find the given value {target_index} in the indices map: {full_indices_map}")
        if target_index in value:
            indices_map_next[key].append(target_index)
            indices_map_next[key] = sorted(indices_map_next[key])

            within_index = indices_map_next[key].index(target_index)
            original_index = full_indices_map[key].index(target_index)
            original_data = original_centroids[i][:, original_index]
            if new_centroids[i].size != 0:
                new_centroids[i] = np.insert(new_centroids[i], within_index, original_data, 1)
            else:
                new_centroids[i] = np.expand_dims(original_data, axis=1)
            if i == 3:
                indices_map_next["time_max_values"].insert(within_index, full_indices_map["time_max_values"][within_index])
            return indices_map_next, new_centroids
        i+=1
        
def transform_to_percent(value):
    """ Transform float value to readable percent value, rounded to the 3rd digit.
    
    :param value:   Float value
    
    :returns readable string representation of the value.
    """
    return str(round(value*100-1, 3))

def remove_feature_from_indices_map(feature_ranking, columns, indices_map, old_centroids):
    """ Removes a feature from the indices map. Adjusts the given centroids accordingly by popping the last feature of the given ranking.
    
    :param feature_ranking:         The ranking of features. Removes the last feature
    :param full_indices_map:        The original full map of indices
    :param columns:                 The full list of feature names
    :param indices_map:             The current indices map (which the feature will be added to)
    :param old_centroids:                 The current centroids


    :returns the updated indices_map and the new centroids
    """
    indices_map_next = deepcopy(indices_map)
    removed_feature = feature_ranking.pop()
    index_value = columns.index(removed_feature)
    new_centroids = deepcopy(old_centroids)

    i = 0
    for key, value_list in indices_map.items():
        # We should find the value before without looking at the time_max_values part of our indices.
        if i > 3:
            raise RuntimeError(f"Didn't expect to not find the given value {index_value} in the indices map: {indices_map}")
        if index_value in value_list:
            internal_index = value_list.index(index_value)
            # Delete column
            if new_centroids[i].shape[1] == 1:
                new_centroids[i] = np.empty(0)
            else:
                new_centroids[i] = np.delete(new_centroids[i], internal_index, 1)
            
            if i == 3:
                indices_map_next["time_max_values"].pop(internal_index)

            

            indices_map_next[key] = [x for x in value_list if x != index_value]
            break
        i+=1
    return indices_map_next, removed_feature, new_centroids
    
def update_centroids(old_centroids, full_indices, sparse_indices):
    """ Updates the given centroids based on the sparse set of indices

    :param old_centroids:   Old centroids to trim down
    :param full_indices:    Original Indices    
    :param sparse_indices:  Target Indices to trim the centroids down to
    
    :returns the updated centroids based on the sparse indices
    """
    new_centroids = deepcopy(old_centroids)
    i=0
    for key, value_list in full_indices.items():
        deletable_indices = []
        for index, v in enumerate(sorted(value_list)):
            if v not in sparse_indices[key]:
                deletable_indices.append(index)
        if len(deletable_indices) > 0:
            new_centroids[i] = np.delete(new_centroids[i], deletable_indices, 1)
        
        i+=1
    return new_centroids

def _split_num_cat(X, numerical, categorical, lists, times):
    """Extract numerical and categorical columns.
    Convert to numpy arrays, if needed.

    :param X: Feature matrix
    :param numerical: Indices of numerical columns
    :param categorical: Indices of categorical columns
    :param lists: Indices of lists columns
    :param times: Indices of time-related columns
    """
    if len(numerical)>0:
        Xnum = np.asanyarray(X[:, numerical]).astype(np.float32)
    else:
        Xnum = np.empty(0)

    if len(categorical)>0:
        Xcat = np.asanyarray(X[:, categorical]).astype(np.int32)
    else:
        Xcat = np.empty(0)

    if len(lists) > 0:
        Xlists = np.asanyarray(X[:, lists])
    else:
        Xlists = np.empty(0)

    if len(times) > 0:
        Xtime = np.asanyarray(X[:, times]).astype(np.int32)
    else:
        Xtime = np.empty(0)

    return Xnum, Xcat, Xlists, Xtime

def split_num_cat(X, mapper):
    """Extract numerical and categorical columns.
    Convert to numpy arrays, if needed.

    :param X: Feature matrix
    :param numerical: Indices of numerical columns
    :param categorical: Indices of categorical columns
    :param lists: Indices of lists columns
    :param times: Indices of time-related columns
    """
    return _split_num_cat(X, mapper.get("num_indices"), mapper.get("cat_indices"), mapper.get("list_indices"), mapper.get("time_indices"))

def initialize_gamma(Xnum, n_cluster):
    """ Initializes gamma values based on the standard deviation within the given dataset of numerical variables (see Huang [1997]).
    
    :param Xnum:        Numerical part of the dataset
    :param n_cluster:   Number of clusters

    :returns array of gamma values
    """
    if Xnum.any():
        return np.array([0.5 * Xnum.std()] * n_cluster, dtype=np.float32)
    else:
        return np.array([1]*n_cluster, dtype=np.int32)


if __name__ == "__main__":
    print(config_to_str())