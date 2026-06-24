import numpy as np

def chi2_independence(C):
    C = np.asarray(C, dtype=np.float64)
    row_total = np.sum(C, axis=1, keepdims=True)
    col_total = np.sum(C, axis=0, keepdims=True)
    total = np.sum(C)
    expected = np.outer(row_total, col_total) / total
    chi2 = np.sum((C - expected)**2 / expected)
    return chi2, expected