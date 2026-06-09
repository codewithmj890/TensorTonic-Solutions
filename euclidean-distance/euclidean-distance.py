import numpy as np

def euclidean_distance(x,y):
    return float(np.sqrt(np.sum((np.asarray(x) - np.asarray(y))**2)))