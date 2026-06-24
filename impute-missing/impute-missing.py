import numpy as np

def impute_missing(X, strategy='mean'):
    X = np.asarray(X, dtype=float)
    result = X.copy()

    # handle 1D case
    if result.ndim == 1:
        result = result.reshape(-1, 1)
        squeeze = True
    else:
        squeeze = False

    for col in range(result.shape[1]):
        column = result[:, col]
        mask = ~np.isnan(column)

        if not np.any(mask):
            result[:, col] = 0
            continue

        if strategy == 'mean':
            fill = np.mean(column[mask])
        else:
            fill = np.median(column[mask])

        column[~mask] = fill

    return result.squeeze() if squeeze else result