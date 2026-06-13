import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    X = np.array(X, dtype=float)
    mu = X.mean(axis=axis, keepdims=True)
    sigma = X.std(axis=axis, keepdims=True)
    return (X - mu) / (sigma + eps)