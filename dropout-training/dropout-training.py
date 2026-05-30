import numpy as np

def dropout(x, p=0.5, rng=None):
    x = np.asarray(x, dtype=float)
    rand = rng.random(x.shape) if rng is not None else np.random.random(x.shape)
    mask = (rand >= p).astype(x.dtype)
    scale = 1.0 / (1.0 - p)
    dropout_pattern = mask * scale
    output = x * dropout_pattern
    return (output, dropout_pattern)