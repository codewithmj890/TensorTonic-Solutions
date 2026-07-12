def priority_replay_sample(priorities, alpha, beta):
    N = len(priorities)

    powered = [p ** alpha for p in priorities]
    total = sum(powered)
    probs = [p / total for p in powered]

    raw_weights = [(N * p) ** (-beta) for p in probs]
    max_w = max(raw_weights)
    norm_weights = [w / max_w for w in raw_weights]

    return [probs, norm_weights]