import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    param = np.asarray(param, dtype=float)
    grad  = np.asarray(grad,  dtype=float)
    m     = np.asarray(m,     dtype=float)
    v     = np.asarray(v,     dtype=float)

    m_new = beta1 * m + (1 - beta1) * grad          # Step 1: first moment
    v_new = beta2 * v + (1 - beta2) * grad ** 2     # Step 2: second moment

    m_hat = m_new / (1 - beta1 ** t)                # Step 3: bias-corrected
    v_hat = v_new / (1 - beta2 ** t)

    param_new = param - lr * m_hat / (np.sqrt(v_hat) + eps)  # Step 4: update

    return (param_new, m_new, v_new)