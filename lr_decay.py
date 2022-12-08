def lr_decay(optimizer, hp):
    """Decay learning rate by a factor of lr_decay"""
    for param_group in optimizer.param_groups:
        if param_group["lr"] > hp.min_lr:
            param_group["lr"] *= hp.lr_decay
    return optimizer
