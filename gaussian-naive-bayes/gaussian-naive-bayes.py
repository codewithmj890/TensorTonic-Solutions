import numpy as np

def gaussian_naive_bayes(X_train, y_train, X_test):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=np.float64)

    classes = np.unique(y_train)
    n = len(y_train)
    eps = 1e-9

    priors = {}
    means = {}
    variances = {}

    for c in classes:
        mask = (y_train == c)
        Xc = X_train[mask]
        nc = Xc.shape[0]

        priors[c] = nc / n
        means[c] = Xc.mean(axis=0)
        variances[c] = Xc.var(axis=0) + eps  # population variance (ddof=0 default)

    predictions = []
    for x in X_test:
        best_class = None
        best_log_post = -np.inf

        for c in classes:
            mu = means[c]
            var = variances[c]

            log_prior = np.log(priors[c])
            log_likelihood = np.sum(
                -0.5 * np.log(2 * np.pi * var) - ((x - mu) ** 2) / (2 * var)
            )

            log_post = log_prior + log_likelihood

            if log_post > best_log_post:
                best_log_post = log_post
                best_class = c

        predictions.append(int(best_class))

    return predictions