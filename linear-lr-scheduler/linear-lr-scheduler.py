def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return step * initial_lr / warmup_steps
    elif step <= total_steps:
        denom = total_steps - warmup_steps
        if denom == 0:
            return final_lr
        return final_lr + (initial_lr - final_lr) * (total_steps - step) / denom
    else:
        return final_lr