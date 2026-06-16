import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    try:
        m = np.asarray(matrix, dtype=np.float64)
        if m.ndim != 2:
            return None
        if axis not in (0, 1, None):
            return None
        if norm_type not in ('l1', 'l2', 'max'):
            return None
    except Exception:
        return None

    if norm_type == 'l1':
        norms = np.sum(np.abs(m), axis=axis, keepdims=True)
    elif norm_type == 'l2':
        norms = np.sqrt(np.sum(m**2, axis=axis, keepdims=True))
    else:  # max
        norms = np.max(np.abs(m), axis=axis, keepdims=True)

    norms = np.where(norms == 0, 1, norms)
    return m / norms