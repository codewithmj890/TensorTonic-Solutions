import numpy as np
def nadam_step(w, m, v, grad, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
    w, m, v, grad = (np.asarray(a, dtype=float) for a in (w, m, v, grad))
    m    = beta1 * m + (1 - beta1) * grad
    v    = beta2 * v + (1 - beta2) * grad**2
    m_nesterov = beta1 * m + (1 - beta1) * grad
    w    = w - lr * m_nesterov / (np.sqrt(v) + eps)
    return w, m, v