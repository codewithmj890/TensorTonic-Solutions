def hit_rate_at_k(recommendations, ground_truth, k):
    if len(recommendations) == 0:
        return 0.0

    hits = 0
    for rec_list , relevant in zip(recommendations, ground_truth):
        top_k = set(rec_list[:k])
        if top_k & set(relevant):
            hits += 1

    return hits / len(recommendations)
        