import numpy as np

def swish(x):
    x = np.asarray(x, dtype = np.float64)
    return x / (1 + np.exp(-x))