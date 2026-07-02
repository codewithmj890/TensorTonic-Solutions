def compute_monitoring_metrics(system_type, y_true, y_pred):
    n = len(y_true)

    if system_type == "classification":
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

        accuracy = (tp + tn) / n if n > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics = [
            ("accuracy", accuracy),
            ("f1", f1),
            ("precision", precision),
            ("recall", recall),
        ]

    elif system_type == "regression":
        errors = [t - p for t, p in zip(y_true, y_pred)]
        mae = sum(abs(e) for e in errors) / n
        rmse = (sum(e ** 2 for e in errors) / n) ** 0.5

        metrics = [
            ("mae", mae),
            ("rmse", rmse),
        ]

    elif system_type == "ranking":
        k = 3
        total_relevant = sum(1 for t in y_true if t == 1)

        pairs = sorted(zip(y_true, y_pred), key=lambda x: x[1], reverse=True)
        top_k = pairs[:k]
        relevant_in_top_k = sum(1 for t, p in top_k if t == 1)

        precision_at_3 = relevant_in_top_k / k if k > 0 else 0.0
        recall_at_3 = relevant_in_top_k / total_relevant if total_relevant > 0 else 0.0

        metrics = [
            ("precision_at_3", precision_at_3),
            ("recall_at_3", recall_at_3),
        ]

    else:
        metrics = []

    return sorted(metrics, key=lambda x: x[0])