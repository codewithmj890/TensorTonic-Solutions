def maxpool_forward(X, pool_size, stride):
    H = len(X)
    W = len(X[0])
    p = pool_size
    s = stride

    H_out = (H - p) // s + 1
    W_out = (W - p) // s + 1

    output = [[0] * W_out for _ in range(H_out)]

    for i in range(H_out):
        for j in range(W_out):
            window_max = X[i * s][j * s]
            for a in range(p):
                for b in range(p):
                    val = X[i * s + a][j * s + b]
                    if val > window_max:
                        window_max = val
            output[i][j] = window_max

    return output