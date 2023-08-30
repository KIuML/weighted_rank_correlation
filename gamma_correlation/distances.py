import numpy as np


def d_sum(x: int, y: int, weights: np.ndarray):
    """
    Pseudo-metric according to eq (15)

    :param x: rank of first elem (one-indexed)
    :param y: rank of second elem (one indexed)
    :param weights: distance weight
    :return:
    """
    lo, hi = np.sort([x, y]) - 1
    return min(1, weights[lo:hi].sum())


def d_max(x: int, y: int, weights: np.ndarray):
    """
    Pseudo-metric according to eq (16)

    :param x: rank of first elem (one-indexed)
    :param y: rank of second elem (one-indexed)
    :param weights: distance weight
    :return:
    """
    lo, hi = np.sort([x, y]) - 1
    return weights[lo:hi].max()


def R(x: int, y: int, d: callable, weights: np.ndarray):
    """
    Antisymmetric version of the distance measure, basically clamping values to 0 when the inputs are inversely ordered

    :param x: rank of first elem (one-indexed)
    :param y: rank of second elem (one-indexed)
    :param d: Distance function
    :param weights: distance weight
    :return: value of the fuzzy order relation R as defined in [1]
    """
    if x < y:
        return d(x, y, weights)
    else:
        return 0
