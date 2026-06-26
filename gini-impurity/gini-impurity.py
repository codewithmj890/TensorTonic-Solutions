import numpy as np

def gini_impurity(y_left, y_right):
    y_left  = np.asarray(y_left, dtype=np.float64)
    y_right = np.asarray(y_right, dtype=np.float64)

    def node_gini(y):
        n = len(y)
        if n == 0:
            return 0.0, 0
        _, counts = np.unique(y, return_counts=True)
        p = counts / n          # class probabilities
        return 1.0 - np.sum(p ** 2), n

    gini_l, n_l = node_gini(y_left)
    gini_r, n_r = node_gini(y_right)

    n_total = n_l + n_r
    if n_total == 0:
        return 0.0

    return (n_l / n_total) * gini_l + (n_r / n_total) * gini_r