def mean_rating_imputation(ratings_matrix, mode):
    num_users = len(ratings_matrix)
    num_items = len(ratings_matrix[0]) if num_users > 0 else 0

    result = [row[:] for row in ratings_matrix]

    if mode == "user":
        for u in range(num_users):
            rated = [r for r in ratings_matrix[u] if r != 0]
            if rated:
                user_mean = sum(rated) / len(rated)
                for i in range(num_items):
                    if result[u][i] == 0:
                        result[u][i] = user_mean
    elif mode == "item":
        for i in range(num_items):
            rated = [ratings_matrix[u][i] for u in range(num_users) if ratings_matrix[u][i] != 0]
            if rated:
                item_mean = sum(rated) / len(rated)
                for u in range(num_users):
                    if result[u][i] == 0:
                        result[u][i] = item_mean

    return result