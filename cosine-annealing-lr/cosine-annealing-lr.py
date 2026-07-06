import math
def cosine_annealing_schedule(base_lr, min_lr, total_steps, current_step):
    # calculate the fraction of total steps completed
    progress = current_step / total_steps
    # apply the half cosine formula
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    # Scale and shift the result to fit between base_lr and min_lr
    lr = min_lr + (base_lr - min_lr) * cosine_decay

    return lr