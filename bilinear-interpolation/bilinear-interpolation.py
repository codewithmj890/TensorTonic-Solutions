import math

def bilinear_resize(image, new_h, new_w):
    H = len(image)
    W = len(image[0])
    
    # Initialize the output grid with zeros
    output = [[0.0 for _ in range(new_w)] for _ in range(new_h)]
    
    for i in range(new_h):
        for j in range(new_w):
            
            # 1. Coordinate Mapping
            # If the new dimension is 1, default the source coordinate to 0 to avoid division by zero
            src_y = (i * (H - 1) / (new_h - 1)) if new_h > 1 else 0.0
            src_x = (j * (W - 1) / (new_w - 1)) if new_w > 1 else 0.0
            
            # 2. Find Integer Bounds (Top-Left coordinate)
            y0 = int(math.floor(src_y))
            x0 = int(math.floor(src_x))
            
            # 3. Calculate Fractional Distances
            dy = src_y - y0
            dx = src_x - x0
            
            # 4. Find Bottom-Right coordinate (clamped to image boundaries)
            y1 = min(y0 + 1, H - 1)
            x1 = min(x0 + 1, W - 1)
            
            # 5. Compute the Interpolated Value
            # Top-Left weight: (1 - dy) * (1 - dx)
            # Bottom-Left weight: dy * (1 - dx)
            # Top-Right weight: (1 - dy) * dx
            # Bottom-Right weight: dy * dx
            
            val = (image[y0][x0] * (1 - dy) * (1 - dx) +
                   image[y1][x0] * dy * (1 - dx) +
                   image[y0][x1] * (1 - dy) * dx +
                   image[y1][x1] * dy * dx)
            
            output[i][j] = val
            
    return output