# weighted_rank_correlation
An implementation of Henzgens and Hüllemeiers weighted rank correlation coefficient scaled gamma, provided by Hoang Cong Thanh @Marmalada1. 


To install the dependencies you can use the associated requirements.txt file:

```
pip install requirements.txt
```

To perform a calculation run the function ```scaled_gamma(x,y)``` with x and y being the two list to compare.

Consider the following minimal example for usage:

```python
from gamma_correlation import gen_weights, gamma_corr

ranking_a = [1, 2, 3, 4, 5]
ranking_b = [5, 4, 3, 2, 1]
w = gen_weights("top", len(ranking_a))

print(gamma_corr(ranking_a, ranking_b, weights=w))
```