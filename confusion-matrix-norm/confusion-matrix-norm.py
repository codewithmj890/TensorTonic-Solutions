import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    y_true = np.asarray(y_true, dtype=np.intp)
    y_pred = np.asarray(y_pred, dtype=np.intp)

    if y_true.size == 0:
        K = num_classes if num_classes is not None else 0
        return np.zeros((K, K), dtype=np.float64 if normalize != 'none' else np.int64)

    K = num_classes if num_classes is not None else int(max(y_true.max(), y_pred.max())) + 1

    if np.any((y_true < 0) | (y_true >= K)) or np.any((y_pred < 0) | (y_pred >= K)):
        raise ValueError(f"Labels must be in range [0, {K-1}]")

    # key trick: flatten 2D (i,j) index into 1D → bincount → reshape
    flat_idx = y_true * K + y_pred
    cm = np.bincount(flat_idx, minlength=K * K).reshape(K, K)

    if normalize == 'none':
        return cm

    cm = cm.astype(np.float64)

    if normalize == 'true':
        # row sums: each row should sum to 1 (recall per class)
        denom = cm.sum(axis=1, keepdims=True)
    elif normalize == 'pred':
        # col sums: each col should sum to 1 (precision per class)
        denom = cm.sum(axis=0, keepdims=True)
    elif normalize == 'all':
        # grand total: entire matrix sums to 1
        denom = cm.sum(keepdims=True)
    else:
        raise ValueError(f"normalize must be 'none','true','pred','all'; got '{normalize}'")

    # avoid division by zero: replace 0 denominators with 1 (0/1 = 0, safe)
    denom = np.where(denom == 0, 1, denom)
    return cm / denom