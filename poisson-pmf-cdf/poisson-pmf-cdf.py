import numpy as np

def poisson_pmf_cdf(lam, k):
    def log_pmf(i):
        log_factorial = np.sum(np.log(np.arange(1, i + 1))) if i > 0 else 0.0
        return -lam + i * np.log(lam) - log_factorial

    pmf = float(np.exp(log_pmf(k)))
    cdf = float(np.sum([np.exp(log_pmf(i)) for i in range(k + 1)]))
    return pmf, cdf