import math

def rolling_std(values, window_size):
    n = len(values)
    k = window_size
    result = []
    
    for i in range(n - k + 1):
        window = values[i:i + k]
        mean = sum(window) / k
        variance = sum((x - mean) ** 2 for x in window) / k
        result.append(math.sqrt(variance))
    
    return result