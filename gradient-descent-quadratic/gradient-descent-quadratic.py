def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    x = x0
    for _ in range(steps):
        grad = 2 * a * x + b      # f'(x) = 2ax + b
        x = x - lr * grad         # x = x - lr * f'(x)
    return float(x)