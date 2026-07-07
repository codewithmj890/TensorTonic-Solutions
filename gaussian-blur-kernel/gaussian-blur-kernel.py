import math

def gaussian_kernel(size, sigma):
    # 1. Edge case for a 1x1 kernel
    if size == 1:
        return [[1.0]]
        
    kernel = [[0.0 for _ in range(size)] for _ in range(size)]
    center = size // 2
    
    # We need to keep track of the sum to normalize later
    weight_sum = 0.0
    
    # 2 & 3. Calculate raw Gaussian weights
    for i in range(size):
        for j in range(size):
            # Calculate distance from the center
            x = j - center
            y = i - center
            
            # Apply the Gaussian function
            exponent = -(x**2 + y**2) / (2 * sigma**2)
            val = math.exp(exponent)
            
            kernel[i][j] = val
            weight_sum += val
            
    # 4. Normalize the kernel so all values sum to 1.0
    for i in range(size):
        for j in range(size):
            kernel[i][j] /= weight_sum
            
    return kernel