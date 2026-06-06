import numpy as np

def cross_entropy_loss(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    N = len(y_true)
    correct_probs = y_pred[np.arange(N), y_true]
    return -np.mean(np.log(correct_probs))