import numpy as np

def kl_divergence(p, q, eps=1e-12):
    p = np.asarray(p, dtype= np.float64)
    q = np.asarray(q, dtype= np.float64)

    mask = p > 0

    return float(np.sum(p[mask] * np.log(p[mask] / (q[mask] + eps))))