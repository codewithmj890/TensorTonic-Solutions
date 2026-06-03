def value_iteration_step(values, transitions, rewards, gamma):
    new_values = []
    for s in range(len(values)):
        q_values = [
            rewards[s][a] + gamma * sum(transitions[s][a][s2] * values[s2] for s2 in range(len(values)))
            for a in range(len(rewards[s]))
        ]
        new_values.append(max(q_values))
    return new_values