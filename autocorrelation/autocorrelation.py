def autocorrelation(series, max_lag):
    n = len(series)
    mean = sum(series) / n
    deviations = [x - mean for x in series]
    gamma_0 = sum(d * d for d in deviations)

    if gamma_0 == 0:
        return [1.0] + [0.0] * max_lag

    result = []
    for k in range(max_lag + 1):
        gamma_k = sum(deviations[t] * deviations[t + k] for t in range(n - k))
        result.append(gamma_k / gamma_0)
    return result