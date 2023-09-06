import numpy as np


def prod(a, b):
    return a * b


def luka(a, b):
    """
    Łukasiewicz t-norm

    :param a:
    :param b:
    :return:
    """
    return np.max(a + b - 1, 0)
