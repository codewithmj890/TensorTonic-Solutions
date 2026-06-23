import numpy as np

def geometric_pmf_mean(k, p):
    k = np.asarray(k, dtype=np.float64)
    pmf = (1 - p) ** (k - 1) * p
    mean = float(1 / p)
    return pmf, mean