import numpy as np

def dice_loss(p, y, eps=1e-8):
    p = np.asarray(p, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()

    intersection = np.sum(p*y)

    dice = (2 * intersection + eps) / (np.sum(p) + np.sum(y) + eps)

    return float(1 - dice)