import numpy as np

def t_test_one_sample(x, mu0):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    return float((mean - mu0) / (std / np.sqrt(n)))