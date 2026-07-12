import numpy as np

def decision_tree_split(X, y):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n, d = X.shape

    def gini(labels):
        if len(labels) == 0:
            return 0.0
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels)
        return 1.0 - np.sum(probs ** 2)

    parent_gini = gini(y)

    best_gain = -1.0
    best_feature = None
    best_threshold = None

    for feature in range(d):
        col = X[:, feature]
        unique_vals = np.unique(col)
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

        for t in thresholds:
            left_mask = col <= t
            right_mask = ~left_mask

            n_left = np.sum(left_mask)
            n_right = np.sum(right_mask)

            if n_left == 0 or n_right == 0:
                continue

            left_gini = gini(y[left_mask])
            right_gini = gini(y[right_mask])

            weighted_gini = (n_left / n) * left_gini + (n_right / n) * right_gini
            gain = parent_gini - weighted_gini

            if gain > best_gain + 1e-12:
                best_gain = gain
                best_feature = feature
                best_threshold = t
            elif abs(gain - best_gain) <= 1e-12:
                # tie-break: smallest feature index, then smallest threshold
                if best_feature is None:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = t
                elif feature < best_feature or (feature == best_feature and t < best_threshold):
                    best_gain = gain
                    best_feature = feature
                    best_threshold = t

    return [best_feature, float(best_threshold)]