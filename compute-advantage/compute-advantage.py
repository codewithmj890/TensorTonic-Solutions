import numpy as np

def compute_advantage(states, rewards, V, gamma):
    
    V = np.asarray(V, dtype=float)

    n = len(rewards)
    returns = np.zeros(n, dtype=float)

    # Compute returns backward
    G = 0.0
    for t in range(n - 1, -1, -1):
        G = rewards[t] + gamma * G
        returns[t] = G

    # Compute advantages
    A = returns - V[np.asarray(states)]

    return A