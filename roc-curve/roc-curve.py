import numpy as np

def roc_curve(y_true, y_score):
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score)

    desc_idx   = np.argsort(y_score, kind='stable')[::-1]
    y_true_s   = y_true[desc_idx]
    thresholds = y_score[desc_idx]

    tp = np.cumsum(y_true_s)
    fp = np.cumsum(1 - y_true_s)

    # Keep last occurrence of each unique threshold
    distinct = np.concatenate([np.diff(thresholds) != 0, [True]])

    tp = tp[distinct]
    fp = fp[distinct]
    thresholds = thresholds[distinct]

    total_pos = tp[-1]
    total_neg = fp[-1]

    tpr = tp / total_pos
    fpr = fp / total_neg

    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    thresholds = np.concatenate([[np.inf], thresholds])

    return fpr, tpr, thresholds