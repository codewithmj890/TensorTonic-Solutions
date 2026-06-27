import numpy as np

def _entropy(y):
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
    y          = np.asarray(y)
    split_mask = np.asarray(split_mask, dtype=bool)

    N  = len(y)
    yL = y[split_mask]        # left  child: where mask is True
    yR = y[~split_mask]       # right child: where mask is False

    # Edge case: one side empty → split is useless
    if yL.size == 0 or yR.size == 0:
        return 0.0

    nL, nR = yL.size, yR.size

    H_parent = _entropy(y)
    H_left   = _entropy(yL)
    H_right  = _entropy(yR)

    weighted_child_entropy = (nL / N) * H_left + (nR / N) * H_right

    return float(H_parent - weighted_child_entropy)
