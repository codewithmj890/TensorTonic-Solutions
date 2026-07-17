def cumulative_returns(returns):
    result = []
    wealth = 1.0
    for r in returns:
        wealth *= (1 + r)
        result.append(wealth - 1.0)
    return result