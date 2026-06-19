import numpy as np

def pearson_correlation(X):
    try:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            return None
        N, D = X.shape
        if N < 2:
            return None

        mu = np.mean(X, axis=0)
        Xc = X - mu
        cov = (Xc.T @ Xc) / (N - 1)

        std = np.sqrt(np.diag(cov))
        outer = np.outer(std, std)

        with np.errstate(invalid='ignore'):
            R = cov / outer

        # Only set diagonal to 1.0 where std is non-zero
        for i in range(D):
            if std[i] != 0:
                R[i, i] = 1.0

        return R

    except (ValueError, TypeError):
        return None