from typing import Union

import numpy as np
import errno, sys
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


def weighter(x, modus):
    """Weight function that assigns a weight to each rank.

    :param data: Ranking data
    :param i: Rank index
    :param column: Column index
    :param modus: Weighing that is applied, should be on of {"top bottom", "top", "bottom", "middle", "top bottom exp"}
    :return: Weight corresponding to the rank at index i
    """
    if x == 1:
        return None
    if modus == 'top bottom':
        if x <= 0.5:
            return 1 - 2 * x
        elif 1 > x > 0.5:
            return 2 * x - 1
    elif modus == 'top':
        if x < 1:
            return 1 - x
    elif modus == 'bottom':
        if x < 1:
            return x
    elif modus == 'middle':
        if x <= 0.5:
            return 2 * x
        elif 1 > x > 0.5:
            return 2 - 2 * x
    elif modus == 'top bottom exp':
        if x < 1:
            return 4 * (x - 0.5) ** 2
    else:
        print('No mode for weighting has been set!')


def data_prep(x, y, weights):
    """
    Prepares data for rank correlation computation

    :param x: _description_
    :param y: _description_
    :param weights: _description_
    :return: _description_
    """
    length = len(x)

    data1 = np.column_stack((rankdata(x), rankdata(y)))
    sort_indices = np.argsort(data1[:, 0])
    data1 = data1[sort_indices]

    data2 = np.column_stack((rankdata(x), rankdata(y)))
    sort_indices2 = np.argsort(data2[:, 1])
    data2 = data2[sort_indices2]

    # Auswahl zwischen eigener Gewichtung, oder einer vordefinierten Gewichtung
    if isinstance(weights, str):
        weight1 = np.ones(length)
        weight2 = np.ones(length)
        if weights != "uniform":
            weight1 = np.array([weighter(data1[i, 0] / length, weights) for i in range(length)])
            weight2 = np.array([weighter(data2[i, 1] / length, weights) for i in range(length)])
        return np.column_stack((data1, weight1, weight2))
    elif len(weights) == (length - 1):
        weights = np.append(weights, np.nan)
        return np.column_stack((data1, weights, weights))
    elif len(weights) and len(weights) != (length - 1):
        print('Length of weight vector is not n-1!')
        sys.exit(errno.EACCES)


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
