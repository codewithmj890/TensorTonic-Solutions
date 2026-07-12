def gae(rewards, values, gamma, lam):
    T = len(rewards)
    advantages = [0.0] * T
    running = 0.0
    for t in range(T - 1, -1, -1):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        running = delta + gamma * lam * running
        advantages[t] = running
    return advantages