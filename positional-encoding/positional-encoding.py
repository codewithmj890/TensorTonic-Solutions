import numpy as np

def positional_encoding(seq_len, d_model, base=10000):
    PE = np.zeros((seq_len, d_model))

    # Sine columns: 0, 2, 4, ... → ceil(d_model/2) columns
    i_sin = np.arange(0, d_model, 2)          # even column indices
    div_sin = np.power(float(base), i_sin / d_model)
    PE[:, 0::2] = np.sin(np.arange(seq_len)[:, None] / div_sin)

    # Cosine columns: 1, 3, 5, ... → floor(d_model/2) columns
    i_cos = np.arange(1, d_model, 2)          # odd column indices
    div_cos = np.power(float(base), (i_cos - 1) / d_model)
    PE[:, 1::2] = np.cos(np.arange(seq_len)[:, None] / div_cos)

    return PE