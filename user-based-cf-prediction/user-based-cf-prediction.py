def user_based_cf_prediction(similarities, ratings):
    weighted_sum = 0.0
    weight_total = 0.0

    for s, r in zip(similarities, ratings):
        if s > 0:
            weighted_sum += s * r
            weight_total += s

    if weight_total == 0:
        return 0.0

    return weighted_sum / weight_total