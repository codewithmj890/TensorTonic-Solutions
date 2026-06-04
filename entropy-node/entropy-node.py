import numpy as np

def entropy_node(y):
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / y.size
    return -np.sum(p * np.log2(p + (p == 0)))  # p==0 mask keeps 0*log(0)=0