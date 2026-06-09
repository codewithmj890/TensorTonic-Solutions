import numpy as np

def global_avg_pool(x):
    if x.ndim == 3:
        return x.mean(axis=(1, 2))
    elif x.ndim == 4:
        return x.mean(axis=(2, 3))
    else:
        raise ValueError(f"Expected 3D (C,H,W) or 4D (N,C,H,W), got {x.ndim}D")