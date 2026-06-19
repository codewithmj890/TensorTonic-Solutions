import numpy as np

def covariance_matrix(X):
    try:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            return None
        N, D = X.shape
        if N < 2:
            return None
        mu = np.mean(X, axis=0)        # shape (D,)
        Xc = X - mu                    # shape (N, D)
        cov = (Xc.T @ Xc) / (N - 1)   # shape (D, D)
        return cov
    except (ValueError, TypeError):
        return None