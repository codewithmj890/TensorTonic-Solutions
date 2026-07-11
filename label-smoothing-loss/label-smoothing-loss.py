import math

def label_smoothing_loss(predictions, target, epsilon):
    K = len(predictions)
    loss = 0.0
    for i in range(K):
        if i == target:
            q_i = (1 - epsilon) + epsilon / K
        else:
            q_i = epsilon / K
        loss -= q_i * math.log(predictions[i])
    return loss