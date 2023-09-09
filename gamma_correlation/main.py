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

    def w_aggregator(idx):
        return weight_agg(weight_vec[slice(*(idx - 1))])

    def calculate_pairwise_comparisons(ranking: np.array) -> np.array:
        """
        :param ranking: 1 × ranklength array of an ordering
        :return: 2 × (ranklength over 2) pairwise weight aggregations.
        """
        # upper triangle matrix to calculate all pairwise comparisons
        triu = np.triu_indices(ranklength, 1)
        pair_indices = np.array(triu)
        # calculate pairwise rank positions
        rank_positions = ranking[pair_indices]

        # calculate weight slices and aggregate, return aij and aji
        # reshape the pairs back into a matrix
        agg_weights_matrix = np.zeros([ranklength, ranklength])
        # first we fill lower triangle with the inverse rank positions
        agg_weights_matrix[triu] = np.apply_along_axis(w_aggregator, 0, np.flipud(rank_positions))
        agg_weights_matrix = agg_weights_matrix.T
        # after transposing we fill the top triangle
        agg_weights_matrix[triu] = np.apply_along_axis(w_aggregator, 0, rank_positions)

        return agg_weights_matrix  # ranklength × ranklength

    # calculate all pairwise comparisons for all rankings. This considers the weights
    pairs_a, pairs_b = np.apply_along_axis(calculate_pairwise_comparisons, 1, rankings)  # ranklength × ranklength

    con = tnorm(pairs_a, pairs_b).sum()
    dis = tnorm(pairs_a, pairs_b.T).sum()

    try:
        return (con - dis) / (con + dis)
    except ZeroDivisionError:
        return np.nan


if __name__ == '__main__':
    first = [1, 3, 2, 4]
    second = [1, 2, 4, 3]

    print(gamma_corr(first, second, weights="top"))
