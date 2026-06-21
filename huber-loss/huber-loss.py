import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    e = np.abs(y_true - y_pred)
    return np.mean(np.where(e <= delta, 0.5 * e**2, delta * (e - 0.5*delta)))