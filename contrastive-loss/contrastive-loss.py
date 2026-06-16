import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(b, dtype=np.float64))
    y = np.asarray(y, dtype=np.float64).ravel()

    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must contain only 0 or 1")

    d = np.linalg.norm(a - b, axis=1)

    loss = y * d**2 + (1 - y) * np.maximum(0, margin - d)**2

    return float(loss.mean() if reduction == "mean" else loss.sum())