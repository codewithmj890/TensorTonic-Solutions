import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    if not documents:
        return np.array([]).reshape(0, 0), []

    # Tokenize
    tokenized = [doc.lower().split() for doc in documents]

    # Build vocabulary
    vocab = sorted(set(word for doc in tokenized for word in doc))
    if not vocab:
        return np.zeros((len(documents), 0)), []

    word2idx = {w: i for i, w in enumerate(vocab)}
    N = len(documents)
    V = len(vocab)

    # TF matrix
    tf = np.zeros((N, V))
    for i, tokens in enumerate(tokenized):
        if not tokens:
            continue
        counts = Counter(tokens)
        for word, cnt in counts.items():
            tf[i, word2idx[word]] = cnt / len(tokens)

    # IDF vector: log(N / df(t)), zero if df(t) == N
    df = np.sum(tf > 0, axis=0)          # shape (V,)
    idf = np.where(df == N, 0.0, np.log(N / df))

    return tf * idf, vocab