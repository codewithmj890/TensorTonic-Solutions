import numpy as np

def knn_distance(X_train, X_test, k):
    X_train = np.array(X_train, dtype=float)
    X_test = np.array(X_test, dtype=float)
    
    # Handle 1D arrays → 2D
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
    
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    
    # Broadcast diff: (n_test, n_train, d)
    diff = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    
    # Euclidean distances: (n_test, n_train)
    distances = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Sort each row → get indices closest first
    sorted_indices = np.argsort(distances, axis=1)
    
    # Handle k > n_train: pad with -1
    if k <= n_train:
        return sorted_indices[:, :k].astype(int)
    else:
        result = np.full((n_test, k), -1, dtype=int)
        result[:, :n_train] = sorted_indices[:, :n_train]
        return result