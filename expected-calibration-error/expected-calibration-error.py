def expected_calibration_error(y_true, y_pred, n_bins):
    n = len(y_true)
    bins = [[] for _ in range(n_bins)]

    for y, p in zip(y_true, y_pred):
        idx = int(p * n_bins)
        if idx == n_bins:
            idx = n_bins - 1
        bins[idx].append((y, p))

    ece = 0.0
    for b in bins:
        if not b:
            continue
        m = len(b)
        acc = sum(y for y, p in b) / m
        conf = sum(p for y, p in b) / m
        ece += (m / n) * abs(acc - conf)

    return ece