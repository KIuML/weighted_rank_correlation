from itertools import combinations
from typing import Union

import numpy as np

from gamma_correlation.distances import d_max, R
from gamma_correlation.tnorms import prod


def gen_weights(mode, len_):
    def cropped_linspace(start, end):
        return np.linspace(start, end, len_ + 1)[1:-1]

    match mode:
        case "top":
            return cropped_linspace(1, 0)
        case "bottom":
            return cropped_linspace(0, 1)
        case "top bottom":
            return np.abs(cropped_linspace(1, -1))
        case "middle":
            return 1 - np.abs(cropped_linspace(1, -1))
        case 'top bottom exp':
            return 4 * (cropped_linspace(0, 1) - 0.5) ** 2
        case _:
            raise AttributeError(f'mode "{mode}" not defined')


def gamma_corr(ranking_a: Union[list, np.ndarray], ranking_b: Union[list, np.ndarray], *,
               weights: Union[np.ndarray, str] = 'uniform', tnorm=prod, d=d_max):
    ranking_a, ranking_b = np.array(ranking_a), np.array(ranking_b)
    n = len(ranking_a)

    if isinstance(weights, str):
        weights = gen_weights(weights, n) if weights != "uniform" else np.ones(n)
    elif len(weights) != (n - 1):
        raise AttributeError('Length of weight vector is not n-1!')

    con, dis = 0, 0
    for i, j in combinations(range(n), 2):
        a_ij = R(ranking_a[i], ranking_a[j], d, weights)
        a_ji = R(ranking_a[j], ranking_a[i], d, weights)
        b_ij = R(ranking_b[i], ranking_b[j], d, weights)
        b_ji = R(ranking_b[j], ranking_b[i], d, weights)

        con += tnorm(a_ij, b_ij) + tnorm(a_ji, b_ji)
        dis += tnorm(a_ij, b_ji) + tnorm(a_ji, b_ij)

    try:
        return (con - dis) / (con + dis)
    except ZeroDivisionError:
        return np.nan


if __name__ == '__main__':
    ranking_a = [1, 2, 3, 4, 5]
    ranking_b = [5, 4, 3, 2, 1]

    print(gamma_corr(ranking_a, ranking_b))
