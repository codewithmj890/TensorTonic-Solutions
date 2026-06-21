import numpy as np

def relu(x):
    x = np.asarray(x, dtype=float)
    return np.maximum(0, x)