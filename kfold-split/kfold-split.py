import numpy as np

def kfold_split(N, k, shuffle=True, rng=None):
    indices = np.arange(N)
    
    if shuffle:
        if rng is not None:
            indices = rng.permutation(indices)
        else:
            np.random.shuffle(indices)
    
    folds = np.array_split(indices, k)
    
    return [
        (np.concatenate([folds[j] for j in range(k) if j != i]), folds[i])
        for i in range(k)
    ]