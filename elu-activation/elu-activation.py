import numpy as np

def elu(x, alpha):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x > 0, x, alpha*(np.exp(x) - 1)).tolist()