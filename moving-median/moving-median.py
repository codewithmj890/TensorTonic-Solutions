def moving_median(values, window_size):
    n = len(values)
    k = window_size
    result = []

    for i in range(n - k + 1):
        window = sorted(values[i:i + k])
        mid = k // 2
        if k % 2 == 1:
            median = float(window[mid])
        else:
            median = (window[mid - 1] + window[mid]) / 2.0
        result.append(median)
    return result