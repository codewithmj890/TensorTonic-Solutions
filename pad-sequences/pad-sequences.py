import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    # Edge case: empty input
    if not seqs:
        return np.array([], dtype=int).reshape(0, 0)
    
    # Step 1: Determine target length
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)
    
    # Step 2: Initialize result array with pad_value
    # Shape: (number of sequences, target length)
    num_seqs = len(seqs)
    result = np.full((num_seqs, max_len), pad_value, dtype=int)
    
    # Step 3: Copy each sequence into result
    for i, seq in enumerate(seqs):
        if seq:  # Non-empty sequence
            # Determine how much to copy (truncate if seq is longer than max_len)
            copy_len = min(len(seq), max_len)
            result[i, :copy_len] = seq[:copy_len]
    
    return result