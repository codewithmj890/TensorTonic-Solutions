import numpy as np

def cohens_kappa(rater1, rater2):
    r1 = np.asarray(rater1)
    r2 = np.asarray(rater2)
    n = len(r1)

    po = np.mean(r1 == r2)

    labels = np.unique(np.concatenate([r1, r2]))
    pe = 0.0
    for k in labels:
        p1 = np.sum(r1 == k) / n
        p2 = np.sum(r2 == k) / n
        pe += p1 * p2

    if pe == 1.0:
        return 1.0

    return float((po - pe) / (1 - pe))