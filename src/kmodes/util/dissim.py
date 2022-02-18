"""
Dissimilarity measures for clustering
"""

import numpy as np
from deprecated import deprecated

# Define globally in order to not redefine on each call
count_entries = np.vectorize(lambda x: len(x), otypes=[np.uint16])

def matching_dissim_lists(a: np.ndarray, b: np.ndarray, **_):
    """
    :param a:       2D Numpy array containing frozenset objects
    :param b:       2D Numpy array containing frozenset objects
    
    :returns distance between arrays

    Calculate the distance between sets A and B through the following formula (essentially Anti-Sparse Hamming):
    Note that the  '-' operation translates to Python's implementation of the difference operation for sets.

    dist(A, B) = max(len(A-B), len(B-A)) / max(len(A), len(B))
    
    note that dist(A, B) = dist(B, A)

    e.g.    dist({"A"}, {"B"}) = max(1, 1) / max(1,1) = 1 (max distance)
            dist({"A", "B"}, {"B"}) = max(1, 0) / max(2, 1) = 0.5 (half distance)
            dist({"A", "B"}, {"C", "D"}) = max(2, 2) / max(2, 2) = 1 (max distance)

            dist({"A"}, {"A", "B", "C", "D", "E"}) = max(0, 4) / max(1,5) -> 0.8 (large distance)

    Generally speaking: dist(small_set, large_set) is smaller than dist(small_set, small_set_2) (excluding)

    """
    
    if a.size==0 or b.size==0:
        return 0
    
    # Calculate unidirectional set difference
    lens1 = count_entries(a-b)
    lens2 = count_entries(b-a)

    # Grab the max of each difference
    difference = np.maximum(lens1, lens2)

    # Count length of each set in each row
    lens_a = count_entries(a)
    lens_b = count_entries(b)

    # Calculate the maximum length
    max_lens = np.maximum(lens_a, lens_b)

    res = difference / max_lens
 
    return np.sum(res, axis=1)

def time_dissim(a: np.ndarray, b: np.ndarray, **kwargs):
    """ Calculates the time difference between different objects. In this case time is treated as a 2D measure (because of the repeating nature of variables like "hour of the day")

    :param a:       2D Numpy array
    :param b:       2D Numpy array

    :returns distance between arrays
    
    """
    if a.size==0 or b.size==0:
        return 0

    time_max_values = kwargs['time_max_values']
    time_max_values = np.array(time_max_values, dtype=np.int32)

    # Calculate the actual\ difference, since time is circular
    sub1 = np.absolute(a-b)
    return np.sum(np.minimum(sub1, time_max_values - sub1) / (time_max_values // 2), axis=1, dtype=np.float32)


def matching_dissim(a, b, **_):
    """Simple matching dissimilarity function
    
    :param a:       2D Numpy array
    :param b:       2D Numpy array

    :returns distance between arrays
    """
    if a.size==0 or b.size==0:
        return 0

    return np.sum(a != b, axis=1)


def euclidean_dissim(a, b, **_):
    """Euclidean distance dissimilarity function
    
    :param a:       2D Numpy array
    :param b:       2D Numpy array

    :returns distance between arrays
    """
    if a.size==0 or b.size==0:
        return 0
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("Missing values detected in numerical columns.")
 
    return np.sum((a - b) ** 2, axis=1)

@deprecated(reason="Not used in my repository")
def jaccard_dissim_binary(a, b, **__):
    """Jaccard dissimilarity function for binary encoded variables"""
    if ((a == 0) | (a == 1)).all() and ((b == 0) | (b == 1)).all():
        numerator = np.sum(np.bitwise_and(a, b), axis=1)
        denominator = np.sum(np.bitwise_or(a, b), axis=1)
        if (denominator == 0).any(0):
            raise ValueError("Insufficient Number of data since union is 0")
        else:
            return 1 - numerator / denominator
    raise ValueError("Missing or non Binary values detected in Binary columns.")

@deprecated(reason="Not used in my repository")
def jaccard_dissim_label(a, b, **__):
    """Jaccard dissimilarity function for label encoded variables"""
    if np.isnan(a.astype('float64')).any() or np.isnan(b.astype('float64')).any():
        raise ValueError("Missing values detected in Numeric columns.")
    intersect_len = np.empty(len(a), dtype=int)
    union_len = np.empty(len(a), dtype=int)
    i = 0
    for row in a:
        intersect_len[i] = len(np.intersect1d(row, b))
        union_len[i] = len(np.unique(row)) + len(np.unique(b)) - intersect_len[i]
        i += 1
    if (union_len == 0).any():
        raise ValueError("Insufficient Number of data since union is 0")
    return 1 - intersect_len / union_len


@deprecated(reason="Not used in my repository")
def ng_dissim(a, b, X=None, membship=None):
    """Ng et al.'s dissimilarity measure, as presented in
    Michael K. Ng, Mark Junjie Li, Joshua Zhexue Huang, and Zengyou He, "On the
    Impact of Dissimilarity Measure in k-Modes Clustering Algorithm", IEEE
    Transactions on Pattern Analysis and Machine Intelligence, Vol. 29, No. 3,
    January, 2007

    This function can potentially speed up training convergence.

    Note that membship must be a rectangular array such that the
    len(membship) = len(a) and len(membship[i]) = X.shape[1]

    In case of missing membship, this function reverts back to
    matching dissimilarity (e.g., when predicting).
    """
    # Without membership, revert to matching dissimilarity
    if membship is None:
        return matching_dissim(a, b)

    def calc_cjr(b, X, memj, idr):
        """Num objects w/ category value x_{i,r} for rth attr in jth cluster"""
        xcids = np.where(memj == 1)
        return float((np.take(X, xcids, axis=0)[0][:, idr] == b[idr]).sum(0))

    def calc_dissim(b, X, memj, idr):
        # Size of jth cluster
        cj = float(np.sum(memj))
        return (1.0 - (calc_cjr(b, X, memj, idr) / cj)) if cj != 0.0 else 0.0

    if len(membship) != a.shape[0] and len(membship[0]) != X.shape[1]:
        raise ValueError("'membship' must be a rectangular array where "
                         "the number of rows in 'membship' equals the "
                         "number of rows in 'a' and the number of "
                         "columns in 'membship' equals the number of rows in 'X'.")

    return np.array([np.array([calc_dissim(b, X, membship[idj], idr)
                               if b[idr] == t else 1.0
                               for idr, t in enumerate(val_a)]).sum(0)
                     for idj, val_a in enumerate(a)])

@deprecated(reason="Unfair comparison because of normalization issues between long and short lists")
def old_matching_dissim_lists(a: np.ndarray, b: np.ndarray, list_max_values: np.array, **_):

    lens1 = count_entries(a-b)
    lens2 = count_entries(b-a)
    intermediate = lens1+lens2
 
    return np.sum(intermediate / list_max_values, axis=1)


@deprecated(reason="Good Idea but is not required. Because of the modulo operation, we can directly ")
def time_dissim_old(a: np.ndarray, b: np.ndarray, **kwargs):
    """ Calculates the time difference between different objects. In this case time is treated as a 2D measure (because of the repeating nature of variables like "hour of the day")
    A simple example illustrating the problem is 'Calculating the difference between the feature "Hour of the day". Two instances 1 (representing 1AM) and 22 (representing 10PM).'.
    Treated as a numerical variable, the distance with Manhatten measurements would be abs(22-1) = 21.

    The numerical distance fails to know that the actual distance between 1AM and 10PM is 3 (hours) and not 21 (hours). Time is cyclic.

    In order to correctly calculate the distance on the circle, simple value shifting.
    The formula for one value pair for 'Hour of the Day' is: dist(A,B) = min(abs(A-B), abs((A+12 % 24) - (B+12 % 24)))
    Note that 24 and 12 represent the length (and half of the length) of the cycle.

    To apply this on the given example:
    A=1, B=22
    -> dist(A,B) = min(abs(22-1), abs((1+12 % 24) - (22+12 % 24)))
                 = min(21, abs(13 - 10))
                 = min(21, 3)
                 = 3

    In a final step the distance is normalized by dividing the value over the total length of the cycle: 
    -> dist_norm(A,B) = dist(A,B) / 24 (in the case of 'Hour of the Day')

    So the distance between 1AM and 10PM is 3 (hours).
    
    This formula works for every variable with cyclic numerical instances (e.g. Hour of the Day, Second of the Day, Day of the Week, etc.) that has a finite max length.

    :param a:       2D Numpy array
    :param b:       2D Numpy array

    :returns distance between arrays
    
    """
    if a.size==0 or b.size==0:
        return 0

    time_max_values = kwargs['time_max_values']
    time_max_values = np.array(time_max_values, dtype=np.int32)

    # Calculate the actual difference, since time is circular
    sub1 = np.absolute(a-b)
    a_modified = (a+time_max_values//2) % time_max_values
    b_modified = (b+time_max_values//2) % time_max_values
    sub2 = np.absolute(a_modified - b_modified)

    return np.sum(np.minimum(sub1, sub2) / (time_max_values // 2), axis=1, dtype=np.float32)
