import numpy as np

def expected_value_discrete(x, p):
    x, p = np.asarray(x, dtype=float), np.asarray(p, dtype=float)
    if x.shape != p.shape:
        raise ValueError("x and p must have the same shape")
    if abs(p.sum() - 1.0) > 1e-6:
        raise ValueError("Probabilities must sum to 1")
    return float(np.dot(x, p))