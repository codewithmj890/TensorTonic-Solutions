import numpy as np

def conv2d(x, W, b):
    N, C_in, H, W_in = x.shape
    C_out, _, KH, KW = W.shape
    
    H_out = H - KH + 1
    W_out = W_in - KW + 1
    
    # Pre-allocate output
    y = np.zeros((N, C_out, H_out, W_out))
    
    for i in range(H_out):
        for j in range(W_out):
            # Extract patch: shape (N, C_in, KH, KW)
            patch = x[:, :, i:i+KH, j:j+KW]
            
            # For each output channel, dot with its kernel + bias
            # W shape: (C_out, C_in, KH, KW)
            # patch shape: (N, C_in, KH, KW)
            # We want output shape: (N, C_out)
            
            # einsum: for each n and c_out, sum over c_in, u, v
            y[:, :, i, j] = np.einsum('nchw,ochw->no', patch, W) + b
    
    return y.astype(float)