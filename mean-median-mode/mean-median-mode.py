import numpy as np
from collections import Counter

def mean_median_mode(x):
    x = np.asarray(x)
    mean = float(np.mean(x))
    median = float(np.median(x))
    freq = Counter(x)
    mode = float(min(k for k, v in freq.items() if v == max(freq.values())))
    return mean, median, mode