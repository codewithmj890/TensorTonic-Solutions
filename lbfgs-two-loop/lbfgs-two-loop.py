def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def lbfgs_direction(grad, s_list, y_list):
    m = len(s_list)
    rho = [1.0 / _dot(y_list[i], s_list[i]) for i in range(m)]

    q = list(grad)
    alpha = [0.0] * m

    for i in range(m - 1, -1, -1):
        alpha[i] = rho[i] * _dot(s_list[i], q)
        q = [q_j - alpha[i] * y_j for q_j, y_j in zip(q, y_list[i])]

    s_last = s_list[m - 1]
    y_last = y_list[m - 1]
    gamma = _dot(s_last, y_last) / _dot(y_last, y_last)
    r = [gamma * q_j for q_j in q]

    for i in range(m):
        beta = rho[i] * _dot(y_list[i], r)
        r = [r_j + s_j * (alpha[i] - beta) for r_j, s_j in zip(r, s_list[i])]

    return [-r_j for r_j in r]