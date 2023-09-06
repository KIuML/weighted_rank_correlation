from itertools import combinations
from typing import Union

import numpy as np

from gamma_correlation.distances import d_max
from gamma_correlation.tnorms import prod
from gamma_correlation.util import gen_weights


def gamma_corr(ranking_a: Union[list, np.ndarray], ranking_b: Union[list, np.ndarray], *,
               weights: np.ndarray = None, tnorm=prod, d=d_max):
    ranking_a, ranking_b = np.array(ranking_a), np.array(ranking_b)
    n = len(ranking_a)

    if weights is None:
        weights = np.ones(n - 1)

    con = dis = 0
    for i, j in combinations(range(n), 2):
        a_ij, b_ij = [d([[r[i]], [r[j]]], weights) if r[i] < r[j] else 0 for r in [ranking_a, ranking_b]]
        a_ji, b_ji = [d([[r[j]], [r[i]]], weights) if r[j] < r[i] else 0 for r in [ranking_a, ranking_b]]

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
