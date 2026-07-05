import math

def ndcg(relevance_scores, k):

    n = len(relevance_scores)
    k = min(k, n)

    def dcg(scores):
        total = 0.0
        for i in range(k):
            rel = scores[i]
            gain = (2 ** rel) - 1
            discount = math.log2(i + 2)  # position i+1, discount log2((i+1)+1)
            total += gain / discount
        return total

    actual_dcg = dcg(relevance_scores)
    ideal_scores = sorted(relevance_scores, reverse=True)
    ideal_dcg = dcg(ideal_scores)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg