import numpy as np

def vector_norm_3d(v):
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 1:
        return float(np.sqrt(np.sum(v**2)))
    return np.sqrt(np.sum(v**2, axis=1))