def retraining_policy(daily_stats, config):
    drift_threshold = config["drift_threshold"]
    performance_threshold = config["performance_threshold"]
    max_staleness = config["max_staleness"]
    cooldown = config["cooldown"]
    retrain_cost = config["retrain_cost"]
    budget = config["budget"]

    days_since_retrain = 0
    last_retrain_day = None  # None means cooldown is initially satisfied
    retrain_days = []

    for stat in daily_stats:
        day = stat["day"]
        drift_score = stat["drift_score"]
        performance = stat["performance"]

        days_since_retrain += 1

        triggered = (
            drift_score > drift_threshold
            or performance < performance_threshold
            or days_since_retrain >= max_staleness
        )

        if triggered:
            cooldown_ok = (last_retrain_day is None) or (day - last_retrain_day >= cooldown)
            budget_ok = budget >= retrain_cost

            if cooldown_ok and budget_ok:
                retrain_days.append(day)
                days_since_retrain = 0
                last_retrain_day = day
                budget -= retrain_cost

    return sorted(retrain_days)