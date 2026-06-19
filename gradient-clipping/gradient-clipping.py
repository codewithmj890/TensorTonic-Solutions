import numpy as np

def clip_gradients(g, max_norm):
    try:
        g = np.asarray(g, dtype=float)
        
        if max_norm <= 0:
            return g.copy()
        
        norm = np.linalg.norm(g)
        
        if norm == 0 or norm <= max_norm:
            return g.copy()
        
        return g * (max_norm / norm)
    
    except (ValueError, TypeError):
        return g