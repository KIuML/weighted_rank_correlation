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

    assert -1 == scaled_gamma(ranking_a, ranking_b)

def test_identical():
    ranking_a = [1, 2, 3, 4, 5]
    assert 1 == scaled_gamma(ranking_a, ranking_a)


def test_identical2():
    ranking_a = [1, 2, 3, 4, 5]
    assert 1 == scaled_gamma(ranking_a, ranking_a,weights=np.random.uniform(size=4).tolist())


def test_1():
    ranking_a = [1, 2, 5, 4, 3]
    ranking_b = [1, 2, 3, 4, 5]

    assert 0.838383 == approx(scaled_gamma(ranking_a, ranking_b, weights=np.linspace(1, .25, 4).tolist()))

def test_2():
    ranking_a = [1, 2, 5, 4, 3]
    ranking_b = np.random.permutation(ranking_a)

    assert -0.10112359550561797 == approx(scaled_gamma(ranking_a, ranking_b, weights=np.linspace(1, .25, 4).tolist()))

def test_3():
    ranking_a = [1, 2, 5, 4, 3]
    ranking_b = np.random.permutation(ranking_a)

    assert -0.4482758620689656 == approx(scaled_gamma(ranking_a, ranking_b, weighting="top bottom"))
