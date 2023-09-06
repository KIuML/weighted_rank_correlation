import numpy as np


def agg_clamped_sum(weights: np.array) -> int:
    """
    Sum of weights slice clamped to the [0,1] interval.

    :param weights: distance weight
    :return:
    """
    return np.minimum(weights.sum(), 1)


def agg_max(weights: np.ndarray) -> int:
    """
    Maximum of weights slice

    :param weights: distance weight
    :return:
    """
    return weights.max()
