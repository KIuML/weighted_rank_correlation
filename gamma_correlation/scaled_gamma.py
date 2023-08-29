from typing import Union

import numpy as np
from scipy.stats import rankdata

from gamma_correlation.distances import d_max
from gamma_correlation.norms import product


def R(data, x, y, column, d):
    """returns value of the fuzzy order relation R as defined in [1]

    [1] Henzgen, Sascha; Hüllermeier, Eyke  (2015): Weighted Rank Correlation: A Flexible Approach Based on Fuzzy Order Relations.
    In: Machine Learning and Knowledge Discovery in Databases. European Conference, ECML PKDD 2015, Porto, Portugal

    :param data: Ranking data
    :param x: Rank x
    :param y: Rank y
    :param column: column index
    :param d: Distance function
    :return: Value of strict fuzzy ordering
    """
    if data[x, column] >= data[y, column]:
        return 0
    else:
        return d(data, x, y, column)


def weighter_arr(mode, len_):
    def tmp(start, end):
        return np.linspace(start, end, len_ + 1)[1:-1]

    match mode:
        case "top":
            w = tmp(1, 0)
        case "bottom":
            w = tmp(0, 1)
        case "top bottom":
            w = np.abs(tmp(1, -1))
        case "middle":
            w = 1 - np.abs(tmp(1, -1))
        case 'top bottom exp':
            w = 4 * (tmp(0, 1) - 0.5) ** 2
        case _:
            raise AttributeError(f'mode "{mode}" not defined')

    return np.append(w, np.nan)


def data_prep(ranking_a, ranking_b, weights):
    """
    Prepares data for rank correlation computation

    :param ranking_a: _description_
    :param ranking_b: _description_
    :param weights: _description_
    :return: _description_
    """
    length = len(ranking_a)

    both_rankings = np.column_stack((rankdata(ranking_a), rankdata(ranking_b)))
    sort_indices = np.argsort(both_rankings[:, 0])
    data1 = both_rankings[sort_indices]

    # Auswahl zwischen eigener Gewichtung, oder einer vordefinierten Gewichtung
    if isinstance(weights, str):
        weight = weighter_arr(weights, length) if weights != "uniform" else np.ones(length)
        return np.column_stack((data1, weight))
    elif len(weights) == (length - 1):
        return np.column_stack((data1, np.append(weights, np.nan)))
    elif len(weights) and len(weights) != (length - 1):
        raise AttributeError('Length of weight vector is not n-1!')


def scaled_gamma(ranking_a, ranking_b, weights: Union[np.ndarray, str] = 'uniform', tnorm=product, d=d_max):
    """
    Calculates the scaled gamma rank correlation coefficient as proposed by Henzgen and Hüllereier [1]

    [1] Henzgen, Sascha; Hüllermeier, Eyke  (2015): Weighted Rank Correlation: A Flexible Approach Based on Fuzzy Order Relations.
    In: Machine Learning and Knowledge Discovery in Databases. European Conference, ECML PKDD 2015, Porto, Portugal

    :param ranking_a: Rank data a
    :param ranking_b: Rank data b
    :return: Scaled gamma
    """

    # Data preprocessing
    data = data_prep(ranking_a, ranking_b, weights)
    # Selection of t-norms and t-conorms
    t_norm, t_conorm = tnorm["norm"], tnorm["conorm"]
    # Initial values 
    con = 0
    dis = 0
    ties = 0

    # Computation of concordance, discordance and ties for each pair
    for i in range(len(data)):
        for z in range((i + 1), len(data)):
            # Account for ties
            if (data[i, 0] == data[z, 0]) or (data[i, 1] == data[z, 1]):
                ties += 1
            else:
                ties += t_conorm(1 - d(data, i, z, 0), 1 - d(data, i, z, 1))
                con += t_norm(R(data, i, z, 0, d), R(data, i, z, 1, d))
                con += t_norm(R(data, z, i, 0, d), R(data, z, i, 1, d))
                dis += t_norm(R(data, i, z, 0, d), R(data, z, i, 1, d))
                dis += t_norm(R(data, z, i, 0, d), R(data, i, z, 1, d))
    if (con == 0) and (dis == 0):
        return np.nan
    else:
        return (con - dis) / (con + dis)


if __name__ == '__main__':
    ranking_a = [1, 2, 3, 4, 5]
    ranking_b = [5, 4, 3, 2, 1]

    print(scaled_gamma(ranking_a, ranking_b))
