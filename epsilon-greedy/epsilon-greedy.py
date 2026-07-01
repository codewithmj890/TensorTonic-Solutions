import numpy as np

def epsilon_greedy(q_values, epsilon, rng=None):
    q_values = np.asarray(q_values, dtype=np.float64)
    n_actions = q_values.shape[0]

    if rng is None:
        rng = np.random

    # draw a uniform random number in [0, 1)
    if rng.random() < epsilon:
        # explore: uniformly random action
        action = rng.integers(0, n_actions) if hasattr(rng, "integers") else rng.randint(0, n_actions)
    else:
        # exploit: greedy action (ties broken by first occurrence, like np.argmax)
        action = np.argmax(q_values)

    return int(action)

