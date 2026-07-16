def polynomial_features(values, degree):
    return [[x ** i for i in range(degree + 1)] for x in values]
           
            