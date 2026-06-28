import numpy as np

def rnn_step_backward(dh, cache):
    x_t, h_prev, h_t, W, U, b = [np.asarray(c) for c in cache]
    dh = np.asarray(dh)

    dz = dh * (1 - h_t ** 2)

    dx_t    = W.T @ dz
    dh_prev = U.T @ dz
    dW      = np.outer(dz, x_t)
    dU      = np.outer(dz, h_prev)
    db      = dz

    return dx_t, dh_prev, dW, dU, db
