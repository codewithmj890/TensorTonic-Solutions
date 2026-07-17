def percent_change(series):
    result = []
    for i in range(1, len(series)):
        prev = series[i - 1]
        if prev == 0:
            result.append(0.0)
        else:
            result.append((series[i] - prev) / prev)
    return result