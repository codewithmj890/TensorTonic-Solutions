import numpy as np 

def manhattan_distance(x,y):
    return float(np.sum(np.abs(np.asarray(x) - np.asarray(y))))