def robust_scaling(values):
    def median(vals):
        s = sorted(vals)
        n = len(s)
        mid = n // 2
        if n % 2 == 1:
            return float(s[mid])
        else:
            return (s[mid - 1] + s[mid]) / 2.0

    n = len(values)
    sorted_vals = sorted(values)
    med = median(sorted_vals)

    if n == 1:
        return [0.0]

    mid = n // 2
    if n % 2 == 0:
        lower_half = sorted_vals[:mid]
        upper_half = sorted_vals[mid:]
    else:
        lower_half = sorted_vals[:mid]
        upper_half = sorted_vals[mid + 1:]

    q1 = median(lower_half)
    q3 = median(upper_half)
    iqr = q3 - q1

    result = []
    for x in values:
        if iqr == 0:
            result.append(x - med)
        else:
            result.append((x - med) / iqr)
    return result