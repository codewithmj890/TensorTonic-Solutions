def popularity_ranking(items, min_votes, global_mean):
    m = min_votes
    C = global_mean
    result = []
    for R, v in items:
        wr = (v / (v + m)) * R + (m / (v + m)) * C
        result.append(wr)
    return result