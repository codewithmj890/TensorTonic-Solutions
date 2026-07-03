def validate_records(records, schema):
    type_map = {
        "int": int,
        "float": (int, float),
        "str": str,
    }

    results = []

    for idx, record in enumerate(records):
        errors = []

        for col_def in schema:
            column = col_def["column"]
            expected_type = col_def["type"]
            nullable = col_def.get("nullable", True)

            if column not in record:
                errors.append(f"{column}: missing")
                continue

            value = record[column]

            if value is None:
                if not nullable:
                    errors.append(f"{column}: null")
                continue

            actual_type = type(value)

            if expected_type == "int":
                type_ok = actual_type is int
            elif expected_type == "float":
                type_ok = actual_type in (int, float)
            elif expected_type == "str":
                type_ok = actual_type is str
            else:
                type_ok = False

            if not type_ok:
                errors.append(f"{column}: expected {expected_type}, got {actual_type.__name__}")
                continue

            min_val = col_def.get("min")
            max_val = col_def.get("max")

            if (min_val is not None or max_val is not None) and isinstance(value, (int, float)) and not isinstance(value, bool):
                if min_val is not None and value < min_val:
                    errors.append(f"{column}: out of range")
                elif max_val is not None and value > max_val:
                    errors.append(f"{column}: out of range")

        results.append((idx, len(errors) == 0, errors))

    return results