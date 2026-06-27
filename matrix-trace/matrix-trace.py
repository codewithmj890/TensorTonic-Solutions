import numpy as np

def matrix_trace(A):
    A = np.asarray(A)
    return float(A[np.arange(A.shape[0]), np.arange(A.shape[0])].sum())
