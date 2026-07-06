def warmup_decay_schedule(base_lr, warmup_steps, total_steps, current_step):
    # Phase 1: Warmup
    if current_step < warmup_steps:
        # Linearly increase from 0 to base_lr
        return base_lr * (current_step / warmup_steps)
        
    # Phase 2: Linear Decay
    else:
        # Linearly decrease from base_lr down to 0
        return base_lr * ((total_steps - current_step) / (total_steps - warmup_steps))