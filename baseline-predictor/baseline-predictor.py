def baseline_predict(ratings_matrix, target_pairs):
    num_users = len(ratings_matrix)
    num_items = len(ratings_matrix[0]) if num_users > 0 else 0

    all_ratings = []
    for row in ratings_matrix:
        for r in row:
            if r != 0:
                all_ratings.append(r)
    mu = sum(all_ratings) / len(all_ratings)

    user_bias = [0.0] * num_users
    for u in range(num_users):
        rated = [r for r in ratings_matrix[u] if r != 0]
        if rated:
            user_mean = sum(rated) / len(rated)
            user_bias[u] = user_mean - mu
        else:
            user_bias[u] = 0.0

    item_bias = [0.0] * num_items
    for i in range(num_items):
        rated = [ratings_matrix[u][i] for u in range(num_users) if ratings_matrix[u][i] != 0]
        if rated:
            item_mean = sum(rated) / len(rated)
            item_bias[i] = item_mean - mu
        else:
            item_bias[i] = 0.0

    predictions = []
    for u, i in target_pairs:
        predictions.append(mu + user_bias[u] + item_bias[i])

    return predictions