def simple_moving_average(values, window_size):
    k = window_size
    n = len(values)
    output = []
    for i in range(n - k + 1):
        window = values[i:i + k]
        output.append(sum(window) / k)
    return output