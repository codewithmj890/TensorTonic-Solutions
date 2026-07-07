import math

def roi_pool(feature_map, rois, output_size):
    pooled_outputs = []
    
    for roi in rois:
        x1, y1, x2, y2 = roi
        roi_h = y2 - y1
        roi_w = x2 - x1
        
        # Initialize an empty grid for this specific ROI
        output_grid = [[0 for _ in range(output_size)] for _ in range(output_size)]
        
        # Iterate over the bins in the target output size
        for i in range(output_size):
            for j in range(output_size):
                
                # 1. Calculate horizontal boundaries for the current bin
                h_start = y1 + math.floor(i * roi_h / output_size)
                h_end = y1 + math.floor((i + 1) * roi_h / output_size)
                
                # 2. Calculate vertical boundaries for the current bin
                w_start = x1 + math.floor(j * roi_w / output_size)
                w_end = x1 + math.floor((j + 1) * roi_w / output_size)
                
                # 3. Ensure the bin covers at least one pixel
                if h_start == h_end:
                    h_end = h_start + 1
                if w_start == w_end:
                    w_end = w_start + 1
                
                # 4. Extract the bin values from the feature map and find the max
                max_val = float('-inf')
                for r in range(h_start, h_end):
                    for c in range(w_start, w_end):
                        if feature_map[r][c] > max_val:
                            max_val = feature_map[r][c]
                
                # Assign the max value to the output grid
                output_grid[i][j] = max_val
                
        # Append the completed grid for this ROI to our final list
        pooled_outputs.append(output_grid)
        
    return pooled_outputs