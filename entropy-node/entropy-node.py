import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()

    # stable log: only take log where prob > 0, avoids log2(0) = -inf
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))

    return entropy