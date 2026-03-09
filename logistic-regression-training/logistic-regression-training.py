import numpy as np

def _sigmoid(z):
    return 1/(1 + np.exp(-z))

def train_logistic_regression(X, y, lr=0.1, steps=500):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    N, D = X.shape

    w = np.zeros(D)
    b = 0.0

    for _ in range(steps):
        p = _sigmoid((X @ w) + b)
        error = p - y
        w -= lr*(X.T@error) / N
        b -= lr*error.mean()

    return w,b


X = [[0],[1],[2],[3]]
y = [0,0,1,1]
w,b = train_logistic_regression(X, y, lr=0.1, steps=100)


