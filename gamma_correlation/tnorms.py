def prod(a, b):
    return a * b


def luka(a, b):
    """
    Łukasiewicz t-norm

    :param a:
    :param b:
    :return:
    """
    return max(a + b - 1, 0)
