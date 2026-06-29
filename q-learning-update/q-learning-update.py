import numpy as np

def q_learning_update(Q, s, a, r, s_next, alpha, gamma):
    Q = np.asarray(Q, dtype=np.float64)
    Q_new = Q.copy()   # Don't modify the original Q-table

    # Compute the TD target
    td_target = r + gamma * np.max(Q[s_next])

    # Compute the TD error
    td_error = td_target - Q[s, a]

    # Update the Q-value
    Q_new[s, a] = Q[s, a] + alpha * td_error

    return Q_new