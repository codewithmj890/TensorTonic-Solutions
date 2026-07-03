def deduplicate(records, key_columns, strategy):
    groups = {}
    order = []

    for record in records:
        key = tuple(record.get(col) for col in key_columns)

        if key not in groups:
            order.append(key)
            groups[key] = []

        groups[key].append(record)

    result = []

    for key in order:
        group = groups[key]

        if strategy == "first":
            selected = group[0]
        elif strategy == "last":
            selected = group[-1]
        elif strategy == "most_complete":
            selected = min(
                group,
                key=lambda r: sum(1 for v in r.values() if v is None)
            )
        else:
            selected = group[0]

        result.append(selected)

    return result