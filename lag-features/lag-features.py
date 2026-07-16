def lag_features(series, lags):
    max_lag = max(lags)
    n = len(series)
    
    return [
        [series[t - lag] for lag in lags]
        for t in range(max_lag, n)
    ]