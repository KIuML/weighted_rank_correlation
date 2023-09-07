import numpy as np


def to_ranking(ordering: np.array) -> np.array:
    return np.argsort(ordering, axis=1) + 1


def to_ordering(ranking: np.array) -> np.array:
    return np.argsort(ranking, axis=1) + 1
