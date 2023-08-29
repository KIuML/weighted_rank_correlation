import numpy as np
from pytest import approx
import pytest

from gamma_correlation import scaled_gamma


@pytest.fixture(autouse=True)
def set_random_seed():
    # seeds any random state in the tests, regardless where is is defined
    np.random.seed(0)


def test_uncorrelated():
    ranking_a = [1, 2, 3, 4, 5]
    ranking_b = [5, 4, 3, 2, 1]

    assert scaled_gamma(ranking_a, ranking_b) == -1


def test_identical():
    ranking_a = [1, 2, 3, 4, 5]
    assert scaled_gamma(ranking_a, ranking_a) == 1


def test_identical2():
    ranking_a = [1, 2, 3, 4, 5]
    assert scaled_gamma(ranking_a, ranking_a, weights=np.random.uniform(size=4)) == 1


def test_1():
    ranking_a = [1, 2, 5, 4, 3]
    ranking_b = [1, 2, 3, 4, 5]

    assert approx(scaled_gamma(ranking_a, ranking_b, weights=np.linspace(1, .25, 4))) == 0.838383


def test_2():
    ranking_a = [1, 2, 5, 4, 3]
    ranking_b = np.random.permutation(ranking_a)

    assert approx(scaled_gamma(ranking_a, ranking_b, weights=np.linspace(1, .25, 4))) == -0.10112359550561797


@pytest.mark.parametrize("func,expected",
                         [("top", -0.5483870967),
                          ("bottom", -0.5483870967),
                          ("top bottom", -0.54838709677),
                          ("middle", -0.5483870967),
                          ("top bottom exp", -0.5483870967)])
def test_weights(func, expected):
    a = np.arange(1, 5)
    ranking_a = np.random.permutation(a)
    ranking_b = np.random.permutation(a)

    assert approx(scaled_gamma(ranking_a, ranking_b, weights="top")) == expected
