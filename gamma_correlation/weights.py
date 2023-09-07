import numpy as np


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


def weight_agg_clamped_sum(weights: np.array) -> int:
    """
    Sum of weights slice clamped to the [0,1] interval.

    :param weights: distance weight
    :return:
    """
    return np.minimum(weights.sum(), 1)


def weight_agg_max(weights: np.ndarray) -> int:
    """
    Maximum of weights slice

    :param weights: distance weight
    :return:
    """
    return weights.max(initial=0)
