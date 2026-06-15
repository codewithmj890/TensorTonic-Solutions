import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    x = np.asarray(x, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    beta = np.asarray(beta, dtype=float)
    
    if x.ndim == 2:
        mean = x.mean(axis=0)
        var = x.var(axis=0)
        x_hat = (x - mean) / np.sqrt(var + eps)
        return gamma * x_hat + beta
    else:  # 4D: (N, C, H, W)
        mean = x.mean(axis=(0, 2, 3), keepdims=True)
        var = x.var(axis=(0, 2, 3), keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + eps)
        return gamma[None, :, None, None] * x_hat + beta[None, :, None, None]