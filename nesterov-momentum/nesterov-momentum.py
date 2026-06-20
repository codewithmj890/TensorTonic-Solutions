import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    w, v, grad = np.asarray(w), np.asarray(v), np.asarray(grad)
    new_v = momentum * v + lr * grad
    new_w = w - new_v
    return (new_w, new_v)