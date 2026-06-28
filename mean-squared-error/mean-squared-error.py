import numpy as np


def mean_squared_error(y_pred, y_true):
    y_pred, y_true = np.asarray(y_pred, dtype=np.float64), np.asarray(y_true, dtype=np.float64)
    if y_pred.shape != y_true.shape:
        return None
    return np.mean((y_pred - y_true) ** 2)
