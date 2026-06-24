import numpy as np

def stratified_split(X, y, test_size=0.2, rng=None):
    X = np.asarray(X)
    y = np.asarray(y)

    train_idx = []
    test_idx = []

    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]

        if rng is not None:
            rng.shuffle(cls_indices)
        else:
            np.random.shuffle(cls_indices)

        n_test = max(1, round(len(cls_indices) * test_size))

        if n_test >= len(cls_indices):
            n_test = len(cls_indices) - 1

        test_idx.extend(cls_indices[:n_test])
        train_idx.extend(cls_indices[n_test:])

    train_idx = np.sort(np.array(train_idx))
    test_idx = np.sort(np.array(test_idx))

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]