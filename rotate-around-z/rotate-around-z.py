import numpy as np

def rotate_around_z(points, theta):
    points = np.asarray(points, dtype=np.float64)

    # Remember whether input was a single point
    single_point = (points.ndim == 1)

    # Convert (3,) -> (1,3)
    if single_point:
        points = points.reshape(1, 3)

    # Compute cosine and sine once
    c = np.cos(theta)
    s = np.sin(theta)

    # Extract coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Apply rotation formulas
    x_new = x * c - y * s
    y_new = x * s + y * c

    # Combine coordinates
    rotated = np.column_stack((x_new, y_new, z))

    # Return same shape as input
    if single_point:
        return rotated.reshape(3,)

    return rotated