import numpy as np


def d_sum(pair: np.array, weights: np.array):
    """
    Sum of weights between lower and higher rank clamped to the [0,1] interval.

    :param x: 1-D array of ranks of first elem (one-indexed)
    :param y: 1-D array of ranks of second elem (one indexed)
    :param weights: distance weight
    :return:
    """
    return np.apply_along_axis(lambda idx: np.minimum(weights[slice(*idx)].sum(), 1), 0, np.array(pair) - 1)


def d_max(pair: np.array, weights: np.ndarray):
    """
    Maximum weight between upper and lower rank.

    :param x: rank of first elem (one-indexed)
    :param y: rank of second elem (one-indexed)
    :param weights: distance weight
    :return:
    """
    return np.apply_along_axis(lambda idx: weights[slice(*idx)].max(), 0, np.array(pair) - 1)
