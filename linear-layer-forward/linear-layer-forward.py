def linear_layer_forward(X, W, b):
    n = len(X)
    d_in = len(X[0])
    d_out = len(W[0])

    Y = [[0.0] * d_out for _ in range(n)]

    for i in range(n):
        for j in range(d_out):
            s = 0.0
            for k in range(d_in):
                s += X[i][k] * W[k][j]
            s += b[j]
            Y[i][j] = s

    return Y