import numpy as np

def normalize_3d(v):
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 1:
        norm = np.sqrt(np.sum(v**2))
        if norm > 1e-10:
            return v / norm
        return np.zeros_like(v)
    else:
        norm = np.sqrt(np.sum(v**2, axis=1, keepdims=True))
        safe = norm > 1e-10
        result = np.zeros_like(v)
        result = np.divide(v, norm, out=result, where=safe)
        return result