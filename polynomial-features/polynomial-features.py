def polynomial_features(values, degree):
    result = []
    for x in values:
        row = []
        for i in range(degree + 1):
            row.append(x ** i)

        result.append(row)

    return result
           
            