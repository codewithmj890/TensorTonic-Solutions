import numpy as np
from collections import Counter

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    N = len(docs)
    if N == 0:
        return np.array([])

    doc_lengths = np.array([len(d) for d in docs], dtype=float)
    avgdl = doc_lengths.mean()

    # df per query term
    scores = np.zeros(N)
    for term in query_tokens:
        df = sum(1 for doc in docs if term in set(doc))
        if df == 0:
            continue
        idf = np.log((N - df + 0.5) / (df + 0.5) + 1)
        tf = np.array([doc.count(term) for doc in docs], dtype=float)
        denom = tf + k1 * (1 - b + b * doc_lengths / avgdl)
        scores += idf * (tf * (k1 + 1)) / denom

    return scores