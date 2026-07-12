def k_means_centroid_update(points, assignments, k):
    if points:
        dim = len(points[0])
    else:
        dim = 0

    sums = [[0.0] * dim for _ in range(k)]
    counts = [0] * k

    for p, a in zip(points, assignments):
        counts[a] += 1
        for d in range(dim):
            sums[a][d] += p[d]

    centroids = []
    for j in range(k):
        if counts[j] == 0:
            centroids.append([0.0] * dim)
        else:
            centroids.append([s / counts[j] for s in sums[j]])

    return centroids