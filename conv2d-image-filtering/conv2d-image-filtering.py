def conv2d(image, kernel, stride=1, padding=0):
    # 1. Get original dimensions
    H = len(image)
    W = len(image[0])
    k_h = len(kernel)
    k_w = len(kernel[0])
    
    # 2. Build the padded image
    p_H = H + 2 * padding
    p_W = W + 2 * padding
    padded = [[0.0] * p_W for _ in range(p_H)]
    
    # Copy original image into the center of the padded matrix
    for i in range(H):
        for j in range(W):
            padded[i + padding][j + padding] = image[i][j]
            
    # 3. Calculate output dimensions
    H_out = ((H + 2 * padding - k_h) // stride) + 1
    W_out = ((W + 2 * padding - k_w) // stride) + 1
    
    # Initialize the output matrix with zeros
    output = [[0.0] * W_out for _ in range(H_out)]
    
    # 4. Perform the convolution operation
    for i in range(H_out):         # Iterate over output rows
        for j in range(W_out):     # Iterate over output columns
            
            # Compute the sum for the current kernel position
            val = 0.0
            for m in range(k_h):       # Iterate over kernel rows
                for n in range(k_w):   # Iterate over kernel columns
                    
                    # Calculate the corresponding coordinate on the padded image
                    img_row = (i * stride) + m
                    img_col = (j * stride) + n
                    
                    # Multiply and accumulate
                    val += padded[img_row][img_col] * kernel[m][n]
            
            # Store the computed sum in the output matrix
            output[i][j] = val
            
    return output