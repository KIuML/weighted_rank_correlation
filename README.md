# Weighted Rank Correlation
An implementation of Henzgens and Hüllermeiers weighted rank correlation coefficient scaled gamma, provided by Hoang Cong Thanh @Marmalada1. 

## Installation

The project can be installed with `poetry`

```bash
poetry add git+https://github.com/KIuML/weighted_rank_correlation.git
```

## Usage

To perform a calculation run the function ```gamma_corr(x,y)``` with x and y being the two rankings to compare.

Consider the following minimal example for usage:

```python
from gamma_correlation import gen_weights, gamma_corr

ranking_a = [1, 2, 3, 4, 5]
ranking_b = [5, 4, 3, 2, 1]

print(gamma_corr(ranking_a, ranking_b, weights="top"))
```

This example computes the gamma correlation coefficient for the predifined _top_ weighting function, that emphasizes changes in the upper part of the rankings. 

Custom weighting functions can be defined in terms of a vector of length _n - 1_. The weight at position _i_ describes the degree to which rank position _i_ and _i + 1_ are distinguished from each other. For example, a linearly inreasing weighting (i.e. emphasizing changes at the bottom of the rankings) can be implemented as follows:

```python
from gamma_correlation import gen_weights, gamma_corr

ranking_a = [1, 2, 3, 4, 5]
ranking_b = [3, 2, 5, 4, 1]
w = np.linspace(1, .25, 4)

print(gamma_corr(ranking_a, ranking_b, weights=w))
```
