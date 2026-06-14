import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = np.sum(y_true == y_pred) / len(y_true)
    classes = np.unique(y_true)

    TP = np.array([np.sum((y_pred == c) & (y_true == c)) for c in classes])
    FP = np.array([np.sum((y_pred == c) & (y_true != c)) for c in classes])
    FN = np.array([np.sum((y_pred != c) & (y_true == c)) for c in classes])
    support = TP + FN

    per_p  = np.where((TP + FP) > 0, TP / (TP + FP), 0.0)
    per_r  = np.where((TP + FN) > 0, TP / (TP + FN), 0.0)
    per_f1 = np.where((per_p + per_r) > 0,
                      2 * per_p * per_r / (per_p + per_r), 0.0)

    if average == "micro":
        tp, fp, fn = TP.sum(), FP.sum(), FN.sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

    elif average == "macro":
        precision = per_p.mean()
        recall    = per_r.mean()
        f1        = per_f1.mean()                    # ✅ avg of per-class F1

    elif average == "weighted":
        weights   = support / support.sum()
        precision = np.sum(weights * per_p)
        recall    = np.sum(weights * per_r)
        f1        = np.sum(weights * per_f1)         # ✅ weighted avg of per-class F1

    elif average == "binary":
        idx       = np.where(classes == pos_label)[0][0]
        precision = per_p[idx]
        recall    = per_r[idx]
        f1        = per_f1[idx]

    return {"accuracy": float(accuracy), "precision": float(precision),
            "recall": float(recall), "f1": float(f1)}