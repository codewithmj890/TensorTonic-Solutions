import bisect

def calibrate_isotonic(cal_labels, cal_probs, new_probs):
    n = len(cal_probs)

    # Step 1: sort calibration data by predicted probability
    idx = sorted(range(n), key=lambda i: cal_probs[i])
    sorted_probs = [cal_probs[i] for i in idx]
    sorted_labels = [cal_labels[i] for i in idx]

    # Step 2: Pool Adjacent Violators Algorithm (PAVA)
    # Each stack element is [sum_of_labels, count]
    stack = []
    for y in sorted_labels:
        stack.append([y, 1])
        while len(stack) > 1 and stack[-2][0] / stack[-2][1] > stack[-1][0] / stack[-1][1]:
            s2, c2 = stack.pop()
            s1, c1 = stack.pop()
            stack.append([s1 + s2, c1 + c2])

    # Expand pooled blocks back into per-point calibrated values
    calibrated = []
    for s, c in stack:
        avg = s / c
        calibrated.extend([avg] * c)

    # Step 3: interpolate new predictions against (sorted_probs, calibrated)
    result = []
    for q in new_probs:
        if q <= sorted_probs[0]:
            result.append(calibrated[0])
        elif q >= sorted_probs[-1]:
            result.append(calibrated[-1])
        else:
            pos = bisect.bisect_left(sorted_probs, q)
            if sorted_probs[pos] == q:
                result.append(calibrated[pos])
            else:
                p_lo = sorted_probs[pos - 1]
                p_hi = sorted_probs[pos]
                c_lo = calibrated[pos - 1]
                c_hi = calibrated[pos]
                val = c_lo + (q - p_lo) / (p_hi - p_lo) * (c_hi - c_lo)
                result.append(val)

    return result