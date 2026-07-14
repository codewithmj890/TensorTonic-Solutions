import math

def perplexity(prob_distributions, actual_tokens):
    n = len(actual_tokens)
    log_probs = [
        math.log(prob_distributions[i][actual_tokens[i]])
        for i in range(n)
    ]
    cross_entropy = -sum(log_probs) / n
    return math.exp(cross_entropy)