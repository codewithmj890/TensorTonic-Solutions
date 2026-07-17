from collections import Counter
def frequency_encoding(values):
    count = Counter(values)
    total = len(values)
    return [count[v] / total for v in values]