"""
Generic utilities for clustering
"""

import numpy as np


def pandas_to_numpy(x):
    return x.values if 'pandas' in str(x.__class__) else x


def get_max_value_key(dic: dict):
    """Gets the key for the maximum value in a dict."""
    if len(dic) == 0:
        return -1

    v = np.array(list(dic.values()))
    k = np.array(list(dic.keys()))
    
    maxima = np.where(v == np.max(v))[0]
    if len(maxima) == 1:
        return k[maxima[0]]
    else:
        # In order to be consistent, always selects the minimum key
        # (guaranteed to be unique) when there are multiple maximum values.
        return k[maxima[np.argmin(k[maxima])]]

def find_max_frequency_attribute(X: np.ndarray, is_list:bool=False) -> int:
    if X.size==0:
        return np.empty(0)
    max_values = np.zeros((X.shape[1]), dtype=object)

    for i in range(max_values.shape[0]):
        if is_list:
            data = np.array([str(v) for v in X[:, i]])
        else:
            data = X[:, i]
        values, counts = np.unique(data, return_counts=True)

        if is_list:
            max_values[i] = eval(values[np.argmax(counts)])
        else:
            max_values[i] = values[np.argmax(counts)]
    return max_values


def encode_features(X, enc_map=None):
    """Converts categorical values in each column of X to integers in the range
    [0, n_unique_values_in_column - 1], if X is not already of integer type.

    If mapping is not provided, it is calculated based on the values in X.

    Unknown values during prediction get a value of -1. np.NaNs are ignored
    during encoding, and get treated as unknowns during prediction.
    """
    if enc_map is None:
        fit = True
        # We will calculate enc_map, so initialize the list of column mappings.
        enc_map = []
    else:
        fit = False

    Xenc = np.zeros(X.shape, dtype='uint32')
    for ii in range(X.shape[1]):
        if fit:
            col_enc = {val: jj for jj, val in enumerate(np.unique(X[:, ii]))
                       if not (isinstance(val, float) and np.isnan(val))}
            enc_map.append(col_enc)
        # Unknown categories (including np.NaNs) all get a value of -1.
        Xenc[:, ii] = np.array([enc_map[ii].get(x, -1) for x in X[:, ii]])

    return Xenc, enc_map


def decode_centroids(encoded, mapping):
    """Decodes the encoded centroids array back to the original data
    labels using a list of mappings.
    """
    decoded = []
    for ii in range(encoded.shape[1]):
        # Invert the mapping so that we can decode.
        inv_mapping = {v: k for k, v in mapping[ii].items()}
        decoded.append(np.vectorize(inv_mapping.__getitem__)(encoded[:, ii]))
    return np.atleast_2d(np.array(decoded)).T


def get_unique_rows(a):
    """Gets the unique rows in a numpy array."""
    return np.vstack(list({tuple(row) for row in a}))
