import numpy as np

def one_hot(y, num_classes=None):
    y = np.asarray(y)
    K = int(np.max(y)) + 1 if num_classes is None else num_classes
    if np.any(y >= K) or np.any(y < 0):
        raise ValueError(f"Labels must be in [0, {K-1}]")
    Y = np.zeros((len(y), K), dtype=float)
    Y[np.arange(len(y)), y] = 1.0
    return Y