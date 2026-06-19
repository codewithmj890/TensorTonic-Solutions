import numpy as np

def calculate_eigenvalues(matrix):
    try:
        mat = np.asarray(matrix, dtype=complex)
        if mat.ndim != 2 or mat.shape[0] == 0 or mat.shape[0] != mat.shape[1]:
            return None
        eigenvalues = np.linalg.eigvals(mat)
        clean = np.round(eigenvalues.real, 8) + 1j * np.round(eigenvalues.imag, 8)
        idx = np.lexsort((clean.imag, clean.real))
        return clean[idx]
    except (ValueError, TypeError):
        return None