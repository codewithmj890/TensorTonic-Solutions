def max_pooling_2d(X, pool_size):
    p = pool_size
    H = len(X)
    W = len(X[0])
    H_out = H // p
    W_out = W // p

    output = []
    for i in range(H_out):
        row = []
        for j in range(W_out):
            window = [
                X[i * p + a][j * p + b]
                for a in range(p)
                for b in range(p)
            ]
            row.append(max(window))
        output.append(row)
    return output