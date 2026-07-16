def interaction_features(X):
    result = []
    for row in X:
        d = len(row)
        interactions = []
        for i in range(d):
            for j in range(i + 1, d):
                interactions.append(row[i] * row[j])
        result.append(list(row) + interactions)
    return result