import numpy as np

def majority_classifier(y_train, X_test):
    y_train = np.asarray(y_train, dtype=np.float64)
    X_test  = np.asarray(X_test, dtype=np.float64)

    classes, counts = np.unique(y_train, return_counts=True)
    majority_class  = classes[np.argmax(counts)]

    return np.full(shape=X_test.shape[0], fill_value=majority_class,dtype=y_train.dtype)