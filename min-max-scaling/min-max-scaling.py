def min_max_scaling(data):
    num_rows = len(data)
    num_cols = len(data[0])

    col_mins = [min(row[j] for row in data) for j in range(num_cols)]
    col_maxs = [max(row[j] for row in data) for j in range(num_cols)]

    scaled = []
    for row in data:
        new_row = []
        for j in range(num_cols):
            col_range = col_maxs[j] - col_mins[j]
            if col_range == 0:
                new_row.append(0.0)
            else:
                new_row.append((row[j] - col_mins[j]) / col_range)
        scaled.append(new_row)

    return scaled