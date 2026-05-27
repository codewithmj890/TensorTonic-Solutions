import numpy as np

def matrix_transpose(A) -> np.ndarray:
    A = np.array(A)
    N, M = A.shape
    result = np.empty((M, N), dtype=A.dtype)
    
    for i in range(N):
        for j in range(M):
            result[j][i] = A[i][j]
    
    return result