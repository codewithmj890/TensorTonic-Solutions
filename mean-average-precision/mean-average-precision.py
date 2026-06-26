import numpy as np

def mean_average_precision(y_true_list, y_score_list, k=None):
    ap_list = []
    
    for y_true, y_score in zip(y_true_list, y_score_list):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_score = np.asarray(y_score, dtype=np.float64)
        
        sorted_idx = np.argsort(y_score)[::-1]
        
        if k is not None:
            sorted_idx = sorted_idx[:k]
        
        y_true_sorted = y_true[sorted_idx]
        
        n_relevant = y_true.sum()
        if n_relevant == 0:
            ap_list.append(0.0)
            continue
        
        # cumulative relevant count at each rank
        cum_rel = np.cumsum(y_true_sorted)
        ranks = np.arange(1, len(y_true_sorted) + 1)
        precision_at_k = cum_rel / ranks
        
        # only sum precisions where item is relevant
        ap = np.sum(precision_at_k * y_true_sorted) / n_relevant
        ap_list.append(ap)
    
    map_value = float(np.mean(ap_list)) if ap_list else 0.0
    return round(map_value, 4), [round(ap, 4) for ap in ap_list]