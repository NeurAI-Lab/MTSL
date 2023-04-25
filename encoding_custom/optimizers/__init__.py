import torch

base_optimizers = {'SGD': torch.optim.SGD, 'Adam': torch.optim.Adam}


def get_optimizer(args, params_to_optimize, cfg):
    optimizer_args = dict(cfg.OPTIMIZER_ARGS)
    kwargs = {'params': params_to_optimize, 'lr': args.lr,
              'weight_decay': args.weight_decay, **optimizer_args}

    optimizer_fn = base_optimizers.get(args.base_optimizer, None)
    if optimizer_fn is None:
        raise ValueError('Unknown optimizer..')
    optimizer = optimizer_fn(**kwargs)

    return optimizer


def get_lr_scheduler(args, optimizer, cfg):
    lr_scheduler_args = dict(cfg.LR_SCHEDULER_ARGS)
    if args.lr_strategy == "poly":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda x: (1 - x / args.epochs) ** 0.9,
            **lr_scheduler_args)
    elif args.lr_strategy == "stepwise":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, args.lr_decay, **lr_scheduler_args)
    else:
        raise ValueError('Unknown lr scheduler..')

    return lr_scheduler
