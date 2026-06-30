def remove_stopwords(tokens, stopwords):
    """
    Returns: list[str] - tokens with stopwords removed (preserve order)
    """
    stopword_set = set(stopwords)
    return [token for token in tokens if token not in stopword_set]