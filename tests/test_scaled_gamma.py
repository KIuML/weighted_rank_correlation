import numpy as np
from pytest import approx
import pytest

from gamma_correlation import gamma_corr, d_sum, d_max, gen_weights


@pytest.fixture(autouse=True)
def set_random_seed():
    # seeds any random state in the tests, regardless where is is defined
    np.random.seed(0)


def test_uncorrelated():
    ranking_a = [1, 2, 3, 4, 5]
    ranking_b = [5, 4, 3, 2, 1]

    assert gamma_corr(ranking_a, ranking_b) == -1


def test_identical():
    ranking_a = [1, 2, 3, 4, 5]
    assert gamma_corr(ranking_a, ranking_a) == 1


def test_identical2():
    ranking_a = [1, 2, 3, 4, 5]
    assert gamma_corr(ranking_a, ranking_a, weights=np.random.uniform(size=4)) == 1


def test_1():
    ranking_a = [1, 2, 5, 4, 3]
    ranking_b = [1, 2, 3, 4, 5]

    assert approx(gamma_corr(ranking_a, ranking_b, weights=np.linspace(1, .25, 4))) == 0.838383


def test_2():
    ranking_a = [1, 2, 5, 4, 3]
    ranking_b = np.random.permutation(ranking_a)

    assert approx(gamma_corr(ranking_a, ranking_b, weights=np.linspace(1, .25, 4))) == -0.10112359550561797


@pytest.mark.parametrize("mode,expected",
                         [("top", -0.5483870967),
                          ("bottom", -0.25),
                          ("top bottom", -0.5),
                          ("middle", -0.25),
                          ("top bottom exp", -0.5)])
def test_weights(mode, expected):
    n = 4
    a = np.arange(n) + 1
    ranking_a = np.random.permutation(a)
    ranking_b = np.random.permutation(a)

    assert approx(gamma_corr(ranking_a, ranking_b, weights=gen_weights(mode, n))) == expected


@pytest.mark.parametrize("func,expected",
                         [(d_sum, -0.5555555555555556),
                          (d_max, -0.5483870967741935)])
def test_dists(func, expected):
    n = 4
    a = np.arange(n) + 1
    ranking_a = np.random.permutation(a)
    ranking_b = np.random.permutation(a)

    assert approx(gamma_corr(ranking_a, ranking_b, weights=gen_weights("top", n), d=func)) == expected


def test_zero_indexed():
    n = 4
    a = np.arange(n)
    ranking_a = np.random.permutation(a)
    ranking_b = np.random.permutation(a)

    assert approx(gamma_corr(ranking_a, ranking_b)) == -0.3333333333333333


def test_float():
    n = 4
    a = np.arange(n).astype("float")
    ranking_a = np.random.permutation(a)
    ranking_b = np.random.permutation(a)

    assert approx(gamma_corr(ranking_a, ranking_b)) == -0.3333333333333333


@pytest.mark.parametrize("ranking_a, ranking_b, expected",
                         [([1,1,2,3], [1,2,2,3], 1.0)])
def test_ties(ranking_a, ranking_b, expected):

    assert approx(gamma_corr(ranking_a, ranking_b)) == expected

def test_disjoint_ranking():

    with pytest.raises(AttributeError):
        gamma_corr([1.5,3.445,4.3], [3,4,1])