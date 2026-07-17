def seasonal_average(series, period):
    result = []
    for p in range(period):
        values = series[p::period]
        result.append(sum(values) / len(values))
    return result