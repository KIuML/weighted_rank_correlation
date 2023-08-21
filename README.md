# weighted_rank_correlation
An implementation of Henzgens and Hüllemeiers weighted rank correlation coefficient scaled gamma, provided by Hoang Cong Thanh @Marmalada1. 


To install the dependencies you can use the associated requirements.txt file:

```
pip install requirements.txt
```
this program can only be run on python versions **below 3.11.0** because of the library **"numba"**.

To perform a calculation run the function ```scaled_gamma(x,y)``` with x and y being the two list to compare.

Consider the following minimal example for usage:

```
from scaled_gamma import scaled_gamma


if __name__ == "__main__":
     ranking_a = [1,2,3,4,5]
     ranking_b = [5,4,3,2,1]

     print(scaled_gamma(ranking_a,ranking_b))
```