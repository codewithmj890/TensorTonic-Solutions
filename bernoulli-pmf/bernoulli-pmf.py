import numpy as np

def bernoulli_pmf_and_moments(x, p):
    x = np.asarray(x, dtype=np.float64)

    pmf = np.where(x==1, p, 1-p)

    return pmf, float(p), float(p*(1-p))