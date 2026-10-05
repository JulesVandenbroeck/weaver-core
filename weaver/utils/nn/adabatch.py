import torch

from ..logger import _logger


def adabatch_accum_steps(epoch, factor=1, every=10, start=None, max_accum=8):
    """
    AdaBatch-style batch size growth schedule (arXiv:1712.02029): the effective batch size is multiplied by `factor`
    every `every` epochs, starting from epoch `start` (defaults to `every`), by accumulating the gradients of
    several batches before each optimizer step.
    :param epoch: current epoch
    :param factor: growth factor of the effective batch size; 1 disables the growth
    :param every: number of epochs between two increases
    :param start: first epoch with an increased batch size; defaults to `every`
    :param max_accum: maximum number of batches accumulated per optimizer step
    :return: number of batches to accumulate per optimizer step (1 = standard training)
    """
    if start is None:
        start = every
    if factor <= 1 or epoch < start:
        return 1
    return min(factor ** ((epoch - start) // every + 1), max_accum)


def _set_lr(group, lr):
    # update a Tensor lr (`--compile-optimizer`) in-place to avoid recompiling the optimizer step
    if isinstance(group["lr"], torch.Tensor):
        group["lr"].fill_(lr)
    else:
        group["lr"] = lr


def _rescale_lr(opt, scheduler, lr_scale):
    """
    Scales the learning rate of all parameter groups by `lr_scale` w.r.t. the lr without AdaBatch.
    The applied scale is stored in the parameter groups (`adabatch_lr_scale`), and therefore saved together with the
    optimizer state, so that resuming the training with `--load-epoch` gives the same learning rate.
    :param opt: optimizer
    :param scheduler: lr scheduler (or None)
    :param lr_scale: scale factor of the learning rate
    """
    if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
        raise NotImplementedError("`--adabatch-lr-exp` is not supported for the `one-cycle` lr scheduler")
    if isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
        # the lr is recomputed from `base_lrs` at every step: rescale it w.r.t. the (never modified) `initial_lr`
        scheduler.base_lrs = [float(group["initial_lr"]) * lr_scale for group in opt.param_groups]
        for group, lr_lambda, base_lr in zip(opt.param_groups, scheduler.lr_lambdas, scheduler.base_lrs):
            _set_lr(group, base_lr * lr_lambda(scheduler.last_epoch))
    else:
        # no scheduler or chainable schedulers (e.g., `flat+decay`): rescale the current lr w.r.t. the previous scale
        for group in opt.param_groups:
            _set_lr(group, float(group["lr"]) * lr_scale / group.get("adabatch_lr_scale", 1.0))
    for group in opt.param_groups:
        group["adabatch_lr_scale"] = lr_scale


def adabatch_setup(args, opt, scheduler, epoch, tb_helper=None):
    """
    Sets up the AdaBatch batch size growth for one training epoch.
    :param args: parsed command line arguments (`--adabatch-*` options)
    :param opt: optimizer
    :param scheduler: lr scheduler (or None)
    :param epoch: current epoch
    :param tb_helper: TensorboardHelper (or None)
    :return: extra keyword arguments for the train function
    """
    if args.adabatch_every < 1 or args.adabatch_max_accum < 1:
        raise ValueError("`--adabatch-every` and `--adabatch-max-accum` must be positive")
    accum_steps = adabatch_accum_steps(
        epoch,
        factor=args.adabatch_factor,
        every=args.adabatch_every,
        start=args.adabatch_start_epoch,
        max_accum=args.adabatch_max_accum,
    )
    if args.adabatch_lr_exp != 0:
        _rescale_lr(opt, scheduler, accum_steps**args.adabatch_lr_exp)

    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size() if distributed else 1
    effective_batch_size = args.batch_size * world_size * accum_steps
    _logger.info(
        "AdaBatch: epoch %d, accumulating %d batches per optimizer step, effective batch size %d"
        % (epoch, accum_steps, effective_batch_size)
    )
    if args.steps_per_epoch is not None and args.steps_per_epoch % accum_steps != 0:
        _logger.warning(
            "AdaBatch: `steps_per_epoch` (%d) is not a multiple of the accumulation steps (%d), "
            "the gradients of the last %d batches of the epoch will be discarded"
            % (args.steps_per_epoch, accum_steps, args.steps_per_epoch % accum_steps)
        )
    if tb_helper:
        tb_helper.write_scalars([("BatchSize/train (epoch)", effective_batch_size, epoch)])
    return {"accum_steps": accum_steps}
