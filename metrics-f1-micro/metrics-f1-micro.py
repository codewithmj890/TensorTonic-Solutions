def f1_micro(y_true, y_pred) -> float:
    tp = sum(t == p for t, p in zip(y_true, y_pred))
    n = len(y_true)
    fp = fn = n - tp
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom != 0 else 0.0