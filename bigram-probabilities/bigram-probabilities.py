def bigram_probabilities(tokens):
    vocab = sorted(set(tokens))
    V = len(vocab)

    # Initialize counts for all pairs to 0 (so counts dict is complete over V x V)
    counts = {(w1, w2): 0 for w1 in vocab for w2 in vocab}

    # Count actual bigrams
    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i + 1]
        counts[(w1, w2)] += 1

    # Compute row totals (sum of counts for each w1 over all w2 in vocab)
    row_totals = {w1: 0 for w1 in vocab}
    for (w1, w2), c in counts.items():
        row_totals[w1] += c

    # Compute add-1 smoothed probabilities
    probs = {}
    for w1 in vocab:
        denom = row_totals[w1] + V
        for w2 in vocab:
            probs[(w1, w2)] = (counts[(w1, w2)] + 1) / denom

    return counts, probs