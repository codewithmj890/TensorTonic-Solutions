import numpy as np

def silhouette_score(X, labels):
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    n = len(X)
    
    # Pairwise euclidean distance matrix (n x n)
    sq = np.sum(X**2, axis=1)
    dist = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))
    
    unique_labels = np.unique(labels)
    a, b = np.zeros(n), np.full(n, np.inf)
    
    for label in unique_labels:
        mask = labels == label
        # a(i): mean dist to same cluster (exclude self)
        a[mask] = dist[np.ix_(mask, mask)].sum(axis=1) / (mask.sum() - 1)
        # b(i): min mean dist to any other cluster
        for other in unique_labels[unique_labels != label]:
            other_mask = labels == other
            mean_dist = dist[np.ix_(mask, other_mask)].mean(axis=1)
            b[mask] = np.minimum(b[mask], mean_dist)
    
    return float(np.mean((b - a) / np.maximum(a, b)))