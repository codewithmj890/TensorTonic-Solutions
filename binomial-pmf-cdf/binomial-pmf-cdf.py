import numpy as np
from scipy.special import comb


def binomial_pmf_cdf(n, p, k):
    pmf = float(comb(n, k, exact=True)*(p**k)*(1 - p)**(n - k))
    cdf = float(sum(comb(n, i, exact=True)*(p**i)*(1 - p)**(n - i) for i in range(k + 1)))
    return pmf, cdf