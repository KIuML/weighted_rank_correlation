import numpy as np
def gen_weights(mode, len_):
    """
    Generate a weighting vector according to predefined modes

    :param mode: Can be one of {'top', 'bottom', 'top bottom', 'middle', 'top bottom exp', 'uniform'}
    :param len_: Length of the rankings (not of the weight vector)
    """
    def cropped_linspace(start, end):
        return np.linspace(start, end, len_ + 1)[1:-1]

    match mode:
        case 'top':
            return cropped_linspace(1, 0)
        case 'bottom':
            return cropped_linspace(0, 1)
        case 'top bottom':
            return np.abs(cropped_linspace(1, -1))
        case 'middle':
            return 1 - np.abs(cropped_linspace(1, -1))
        case 'top bottom exp':
            return 4 * (cropped_linspace(0, 1) - 0.5) ** 2
        case 'uniform':
            return np.ones(len_-1)
        case _:
            raise AttributeError(f'mode "{mode}" not defined')