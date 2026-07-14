import math
from collections import Counter

def bleu_score(candidate, reference, max_n):
    c = len(candidate)
    r = len(reference)

    if c == 0:
        return 0.0

    def ngrams(tokens, n):
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    log_precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = ngrams(candidate, n)
        if not cand_ngrams:
            return 0.0

        cand_counts = Counter(cand_ngrams)
        ref_counts = Counter(ngrams(reference, n))

        clipped = sum(min(count, ref_counts[ng]) for ng, count in cand_counts.items())
        total = sum(cand_counts.values())

        pn = clipped / total
        if pn == 0:
            return 0.0
        log_precisions.append(math.log(pn))

    if c >= r:
        bp = 1.0
    else:
        bp = math.exp(1 - r / c)

    geo_mean = math.exp(sum(log_precisions) / max_n)
    return bp * geo_mean