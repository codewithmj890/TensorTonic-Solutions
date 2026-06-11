import numpy as np

def focal_loss(p, y, gamma=2.0):
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    p_t = np.where(y == 1, p, 1 - p)
    loss = -((1 - p_t) ** gamma) * np.log(p_t)
    return loss.mean()
    