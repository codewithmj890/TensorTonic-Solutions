import numpy as np

def sample_var_std(x):
    x = np.asarray(x, dtype=float)  
    n = x.size
    mean_x = np.mean(x)
    
    var = np.sum((x - mean_x) ** 2) / (n - 1)  
    std = np.sqrt(var)
    
    return float(var), float(std)