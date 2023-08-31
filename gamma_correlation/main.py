from itertools import combinations
from typing import Union

import numpy as np

from gamma_correlation.util import gen_weights
from gamma_correlation.distances import d_max, R
from gamma_correlation.tnorms import prod

from scipy.stats import rankdata


def gamma_corr(ranking_a: Union[list, np.ndarray], ranking_b: Union[list, np.ndarray], *,
               weights: np.ndarray = None, tnorm=prod, d=d_max):
    """
    Computes the scaled gamma rank correlation.

    :param ranking_a: _description_
    :param ranking_b: _description_
    :param weights: _description_, defaults to None
    :param tnorm: _description_, defaults to prod
    :param d: _description_, defaults to d_max
    :return: _description_
    """
    if set(ranking_a) != set(ranking_b):
        raise AttributeError(f"Ranking a {ranking_a} and ranking b {ranking_b} contain different ranks!")
    ranking_a, ranking_b = rankdata(np.array([ranking_a, ranking_b]), method="dense", axis=1)
    n = len(ranking_a)

    if weights is None:
        weights = np.ones(n - 1)

    con = dis = 0
    for i, j in combinations(range(n), 2):
        a_ij, b_ij = [R(r[i], r[j], d, weights) for r in [ranking_a, ranking_b]]
        a_ji, b_ji = [R(r[j], r[i], d, weights) for r in [ranking_a, ranking_b]]

        con += tnorm(a_ij, b_ij) + tnorm(a_ji, b_ji)
        dis += tnorm(a_ij, b_ji) + tnorm(a_ji, b_ij)

    try:
        return (con - dis) / (con + dis)
    except ZeroDivisionError:
        return np.nan


if __name__ == '__main__':
    ranking_a = [1, 2, 3, 4, 5]
    ranking_b = [5, 4, 3, 2, 1]
    w = gen_weights("top", len(ranking_a))

    print(gamma_corr(ranking_a, ranking_b, weights=w))
