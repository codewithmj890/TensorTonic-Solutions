def winsorize(values, lower_pct, upper_pct):
    def percentile(sorted_vals, p):
        n = len(sorted_vals)
        if n == 1:
            return float(sorted_vals[0])
        k = (n - 1) * p / 100.0
        f = int(k)  # floor
        c = f + 1 if f < n - 1 else f
        frac = k - f
        return sorted_vals[f] + frac * (sorted_vals[c] - sorted_vals[f])

    sorted_vals = sorted(values)
    lower_bound = percentile(sorted_vals, lower_pct)
    upper_bound = percentile(sorted_vals, upper_pct)

    result = []
    for x in values:
        if x < lower_bound:
            result.append(lower_bound)
        elif x > upper_bound:
            result.append(upper_bound)
        else:
            result.append(float(x))
    return result