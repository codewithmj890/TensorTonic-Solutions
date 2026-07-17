def adjusted_cosine_similarity(ratings_matrix, item_i, item_j):
    numerator = 0.0
    denom_i = 0.0
    denom_j = 0.0

    for user_ratings in ratings_matrix:
        r_i = user_ratings[item_i]
        r_j = user_ratings[item_j]

        if r_i == 0 or r_j == 0:
            continue

        rated_items = [r for r in user_ratings if r != 0]
        user_mean = sum(rated_items) / len(rated_items)

        diff_i = r_i - user_mean
        diff_j = r_j - user_mean

        numerator += diff_i * diff_j
        denom_i += diff_i ** 2
        denom_j += diff_j ** 2

    denominator = (denom_i ** 0.5) * (denom_j ** 0.5)

    if denominator == 0:
        return 0.0

    return numerator / denominator