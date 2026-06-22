import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    a = np.atleast_2d(np.asarray(anchor, dtype = np.float64))
    p = np.atleast_2d(np.asarray(positive, dtype = np.float64))
    n = np.atleast_2d(np.asarray(negative, dtype = np.float64))

    d_ap = np.sum((a - p)**2, axis=1)
    d_an = np.sum((a - n)**2, axis=1)

    return float(np.mean(np.maximum(0, (d_ap - d_an + margin))))