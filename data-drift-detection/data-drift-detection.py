def detect_drift(reference_counts, production_counts, threshold):
    ref_total = sum(reference_counts)
    prod_total = sum(production_counts)

    ref_dist = [c / ref_total for c in reference_counts]
    prod_dist = [c / prod_total for c in production_counts]

    tvd = 0.5 * sum(abs(p - q) for p, q in zip(ref_dist, prod_dist))

    return {"score": tvd, "drift_detected": tvd > threshold}