from operator import xor
from typing import Union, Optional

import numpy as np
from scipy.stats import rankdata

from gamma_correlation.tnorms import prod
from gamma_correlation.weights import gen_weights, weight_agg_max


def gamma_corr(ranking_a: Union[list, np.ndarray], ranking_b: Union[list, np.ndarray], *,
               weights: Optional[Union[str, np.array]] = None, tnorm=prod, weight_agg=weight_agg_max):
    rankings = np.array([ranking_a, ranking_b])
    if not np.array_equal(rankdata(rankings, axis=1, method="ordinal"), rankings):
        raise ValueError("The provided rankings appear to be not proper rankings. Maybe they contain Ties?")
    n, ranklength = rankings.shape

    if weights is None:
        weights = "uniform"
    if isinstance(weights, str):
        weight_vec = gen_weights(weights, ranklength)
    elif isinstance(weights, np.ndarray):
        weight_vec = weights  # type:np.array

    def calculate_pairwise_comparisons(ranking: np.array) -> np.array:
        """
        :param ranking: 1 × n array of an ordering
        :return: 2 × (n over 2) pairwise weight aggregations.
        """
        # upper triangle matrix to calculate all pairwise comparisons
        pair_indices = np.array(np.triu_indices(ranklength, 1))
        # calculate pairwise rank positions
        rank_positions = ranking[pair_indices]
        # calculate weight slices and aggregate, return aij and aji
        agg_weights = np.vstack((
            np.apply_along_axis(lambda idx: weight_agg(weight_vec[slice(*(idx - 1))]), 0, rank_positions),
            np.apply_along_axis(lambda idx: weight_agg(weight_vec[slice(*(idx - 1))]), 0, np.flipud(rank_positions))
        ))

        return agg_weights  # 2 × (ranklength over 2)

    # calculate all pairwise comparisons for all orderings
    pairs_a, pairs_b = np.apply_along_axis(calculate_pairwise_comparisons, 1, rankings)  # 2 × 2 × (ranklength over 2)

    correlations = [((-1) ** xor(x, y)) * tnorm(pairs_a[x], pairs_b[y]) for x, y in [[0, 0], [0, 1], [1, 0], [1, 1]]]
    corr = np.array(correlations)

    try:
        return corr.sum() / np.abs(corr).sum()
    except ZeroDivisionError:
        return np.nan


if __name__ == '__main__':
    first = [1, 3, 2, 4]
    second = [1, 2, 4, 3]

    print(gamma_corr(first, second, weights="top"))
