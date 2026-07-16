def weighted_moving_average(values, weights):
    k = len(weights)
    n = len(values)
    weight_sum = sum(weights)
    
    return [
        sum(w * values[i + j] for j, w in enumerate(weights)) / weight_sum
        for i in range(n - k + 1)
    ]