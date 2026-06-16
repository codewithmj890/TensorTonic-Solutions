import numpy as np
from scipy.special import erf

def gelu(x):
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))
