def jaccard_similarity(set_a, set_b):
    a = set(set_a)
    b = set(set_b)

    if not a and not b:
        return 0.0

    intersection = len(a & b)
    union = len(a | b)

    return intersection / union