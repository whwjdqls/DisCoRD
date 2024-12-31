import math


def adjust_learning_rate(optimizer, epoch, lr, warmup_epochs, min_lr, lr_schedule, epochs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr = lr * epoch / warmup_epochs 
    else:
        if lr_schedule == "constant":
            lr = lr
        elif lr_schedule == "cosine":
            lr = min_lr + (lr - min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
        else:
            raise NotImplementedError
    
    for param_group in optimizer['param_groups']:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
            
    return lr
