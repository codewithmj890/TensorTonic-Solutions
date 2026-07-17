def rank_transform(values):
    n = len(values)
    sorted_indices = sorted(range(n), key=lambda i: values[i])

    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and values[sorted_indices[j]] == values[sorted_indices[i]]:
            j += 1
        # positions i..j-1 (0-based) share the same value
        avg_rank = (i + 1 + j) / 2.0  # average of (i+1) .. j
        for k in range(i, j):
            ranks[sorted_indices[k]] = avg_rank
        i = j

    return ranks