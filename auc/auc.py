import numpy as np

def auc(fpr, tpr):
    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)
    return float(np.trapezoid(tpr, fpr))