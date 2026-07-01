import numpy as np

def detect_skew(train_dist, serving_dist, threshold=0.2, eps=1e-10):
    result = {}
    for feature, train_bins in train_dist.items():
        serving_bins = serving_dist[feature]

        train_p = np.asarray(train_bins, dtype=float) + eps
        serve_p = np.asarray(serving_bins, dtype=float) + eps

        psi = float(np.sum((serve_p - train_p) * np.log(serve_p / train_p)))

        result[feature] = {
            "psi": psi,
            "skewed": psi >= threshold
        }

    return result