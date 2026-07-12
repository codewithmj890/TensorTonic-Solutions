import numpy as np

def random_forest_vote(predictions):
    predictions = np.asarray(predictions)
    T, N = predictions.shape

    results = []
    for i in range(N):
        votes = predictions[:, i]
        counts = np.bincount(votes)
        max_count = counts.max()
        # smallest label among those with max_count
        winner = np.argmax(counts == max_count)
        results.append(int(winner))

    return results