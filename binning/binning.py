def binning(values, num_bins):
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val

    if range_val == 0:
        return [0] * len(values)

    width = range_val / num_bins

    bins = []
    for x in values:
        b = int((x - min_val) // width)
        if b > num_bins - 1:
            b = num_bins - 1
        bins.append(b)

    return bins