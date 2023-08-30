import numpy as np


def d_max(pair: np.ndarray, weights: np.ndarray):
    """Pseudo-metric according to eq (11) in the paper

    :param data: Rank data
    :param i: Rank index i
    :param j: Rank index j
    :param column: column index
    :return: Value of pseudo-metric as a global distance funcion
    """
    low_idx, high_idx = sorted(pair.astype(int) - 1)
    return weights[low_idx:high_idx].max()


def d_sum(pair: np.ndarray, weights: np.ndarray):
    """Pseudo-metric according to eq (12) in the paper

    :param data: Rank data
    :param i: Rank index i
    :param j: Rank index j
    :param column: Column index
    :return: Value of pseudo-metric as a global distance funcion on
    """
    low_idx, high_idx = sorted(pair.astype(int) - 1)
    return min(1, weights[low_idx:high_idx].sum())
