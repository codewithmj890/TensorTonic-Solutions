import numpy as np

def pca_projection(X, k):
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape

    mean = X.mean(axis=0)
    Xc = X - mean

    C = (Xc.T @ Xc) / (n - 1)

    eigvals, eigvecs = np.linalg.eigh(C)  # ascending order, orthonormal columns

    # sort descending by eigenvalue
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    W = eigvecs[:, :k].copy()

    # sign convention: largest-magnitude component of each eigenvector is positive
    for j in range(k):
        col = W[:, j]
        idx = np.argmax(np.abs(col))
        if col[idx] < 0:
            W[:, j] = -col

    X_proj = Xc @ W

    return X_proj.tolist()