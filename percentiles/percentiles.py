import numpy as np

def percentiles(x, q):
    
    x = np.asarray(x, dtype=float)
    q = np.asarray(q, dtype=float)
    
    x = np.sort(x)  
    
    result = np.percentile(x, q, method='linear')
    
    return result