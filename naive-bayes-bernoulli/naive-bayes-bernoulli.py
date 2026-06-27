import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train)
    X_test  = np.array(X_test, dtype=float)

    classes = np.unique(y_train)          # sorted ascending e.g. [0, 1, 2]
    n_train = X_train.shape[0]
    n_classes = len(classes)

    log_priors     = np.zeros(n_classes)         # shape: (n_classes,)
    log_theta      = np.zeros((n_classes, X_train.shape[1]))  # log P(xi=1 | y)
    log_1_m_theta  = np.zeros((n_classes, X_train.shape[1]))  # log P(xi=0 | y)

    for idx, c in enumerate(classes):
        X_c = X_train[y_train == c]      # all training rows belonging to class c
        n_c = X_c.shape[0]              # number of samples in class c

        log_priors[idx] = np.log(n_c / n_train)

        theta = (X_c.sum(axis=0) + 1) / (n_c + 2)   # shape: (d,)

        log_theta[idx]     = np.log(theta)
        log_1_m_theta[idx] = np.log(1 - theta)


    log_likelihood = (X_test @ log_theta.T) + ((1 - X_test) @ log_1_m_theta.T)

    log_posterior = log_likelihood + log_priors

    return log_posterior