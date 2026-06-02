import numpy as np

def apply_homogeneous_transform(T, points):
    points = np.array(points)
    single = points.ndim == 1
    if single:
        points = points[np.newaxis, :]          # (1, 3)
    
    ones = np.ones((points.shape[0], 1))
    ph = np.hstack([points, ones])              # (N, 4)
    transformed = (T @ ph.T).T                  # (N, 4)
    result = transformed[:, :3]                 # (N, 3)
    
    return result[0] if single else result