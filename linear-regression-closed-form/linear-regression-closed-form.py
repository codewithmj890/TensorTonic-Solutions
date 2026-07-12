import numpy as np

def linear_regression_closed_form(X, y):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    Xt = X.T
    XtX = Xt @ X
    Xty = Xt @ y

    w = np.linalg.inv(XtX) @ Xty

    return w.tolist()