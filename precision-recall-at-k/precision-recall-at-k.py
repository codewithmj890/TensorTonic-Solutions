def precision_recall_at_k(recommended, relevant, k):
    top_k = set(recommended[:k])
    relevant_set = set(relevant)
    hits = len(top_k & relevant_set)
    return [hits / k, hits / len(relevant_set)]