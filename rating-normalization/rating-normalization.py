def rating_normalization(matrix):
    result = []
    for row in matrix:
        rated = [v for v in row if v != 0]
        if not rated:
            result.append([0.0 for _ in row])
            continue
        mean = sum(rated) / len(rated)
        result.append([(v - mean) if v != 0 else 0.0 for v in row])
    return result