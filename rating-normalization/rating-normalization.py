def rating_normalization(matrix):
    return [
        [v - sum(rated) / len(rated) if v != 0 else 0.0 for v in row]
        if (rated := [x for x in row if x != 0]) else [0.0] * len(row)
        for row in matrix
    ]