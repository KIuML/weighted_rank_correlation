def d_max(data, i, j, column):
    """Pseudo-metric according to eq (11) in the paper

    :param data: Rank data
    :param i: Rank index i
    :param j: Rank index j
    :param column: column index
    :return: Value of pseudo-metric as a global distance funcion
    """
    return max(data[int(min(data[i, column], data[j, column])) - 1:
                    int(max(data[i, column], data[j, column])) - 1,
               -1])


def d_sum(data, i, j, column):
    """Pseudo-metric according to eq (12) in the paper

    :param data: Rank data
    :param i: Rank index i
    :param j: Rank index j
    :param column: Column index
    :return: Value of pseudo-metric as a global distance funcion on
    """
    return min(1,
               sum(data[int(min(data[i, column], data[j, column])) - 1:
                        int(max(data[i, column], data[j, column])) - 1,
                   -1]))
