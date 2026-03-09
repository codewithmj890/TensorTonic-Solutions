import numpy as np

def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1/(1 + np.exp(-x))

print(sigmoid([0,2,-2]))
print(sigmoid([0]))
print(sigmoid([[-1,0],[1,2]]))