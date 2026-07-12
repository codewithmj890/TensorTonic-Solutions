import numpy as np

def policy_gradient_loss(log_probs, rewards, gamma):
    log_probs = np.asarray(log_probs, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    T = len(rewards)

    returns = np.zeros(T, dtype=np.float64)
    running = 0.0
    for t in range(T - 1, -1, -1):
        running = rewards[t] + gamma * running
        returns[t] = running

    baseline = np.mean(returns)
    advantages = returns - baseline

    loss = -np.mean(log_probs * advantages)

    return float(loss)