def discount_returns(rewards, gamma):
    T = len(rewards)
    G = [0.0] * T
    G[T - 1] = rewards[T - 1]
    for t in range(T - 2, -1, -1):
        G[t] = rewards[t] + gamma * G[t + 1]
    return G