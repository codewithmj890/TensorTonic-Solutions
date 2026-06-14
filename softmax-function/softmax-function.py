import numpy as np

def softmax(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x_shifted = x - np.max(x)
        e_x = np.exp(x_shifted)
        return e_x / np.sum(e_x)
    else:
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        e_x = np.exp(x_shifted)
        return e_x / np.sum(e_x, axis=1, keepdims=True)