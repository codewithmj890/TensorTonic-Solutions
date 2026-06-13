import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    X = np.array(X, dtype=float)
    mn = X.min(axis=axis, keepdims=True)
    mx = X.max(axis=axis, keepdims=True)
    return (X - mn) / (mx - mn + eps)