def histogram_equalize(image):
    hist = [0] * 256
    total_pixels = 0
    for row in image:
        for v in row:
            hist[v] += 1
            total_pixels += 1

    cdf = [0] * 256
    running = 0
    for i in range(256):
        running += hist[i]
        cdf[i] = running

    cdf_min = None
    for i in range(256):
        if cdf[i] != 0:
            cdf_min = cdf[i]
            break

    denom = total_pixels - cdf_min

    if denom == 0:
        return [[0 for _ in row] for row in image]

    def map_val(v):
        return round((cdf[v] - cdf_min) / denom * 255)

    return [[map_val(v) for v in row] for row in image]