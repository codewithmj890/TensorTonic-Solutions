import numpy as np

def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    X = np.asarray(X)
    y = np.asarray(y)

    indices = np.arange(len(X))

    if rng is not None:
        rng.shuffle(indices)
    else:
        np.random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]

        if drop_last and len(batch_idx) < batch_size:
            return

        yield X[batch_idx], y[batch_idx]