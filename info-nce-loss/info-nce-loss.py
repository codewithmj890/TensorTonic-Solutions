import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    Z1 = np.asarray(Z1, dtype = np.float64)
    Z2 = np.asarray(Z2, dtype = np.float64)

    S = np.dot(Z1, Z2.T) / temperature

    S_stable = S - np.max(S, axis=1, keepdims=True)
    exp_S = np.exp(S_stable)

    log_probs = np.log(np.diag(exp_S) / np.sum(exp_S, axis = 1))

    return float(-np.mean(log_probs))