import numpy as np

def ridge_regression(X, y, lam):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    d = X.shape[1]

    XtX = X.T @ X
    reg = XtX + lam * np.eye(d)
    Xty = X.T @ y

    w = np.linalg.inv(reg) @ Xty

    return w.tolist()