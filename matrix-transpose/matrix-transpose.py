import numpy as np

def matrix_transpose(A) -> np.ndarray:
    A = np.array(A)
    N, M = A.shape
    result = np.empty((M, N), dtype=A.dtype)
    
    rows = np.arange(N).reshape(1, N)  # shape (1, N)
    cols = np.arange(M).reshape(M, 1)  # shape (M, 1)
    
    result[cols, rows] = A[rows, cols]  # broadcasting → (M, N)
    
    return result