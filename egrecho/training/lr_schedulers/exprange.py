# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-09)

from functools import partial
from typing import Union

from torch.optim.lr_scheduler import LambdaLR

from .warmup import WARM_LRSCHEDULERS, WarmupHoldScheduler


def exp_scale_fn(cur_step, initial_lr, total_steps, eta_min=1e-6):
    return (eta_min / initial_lr) ** (cur_step / total_steps)


@WARM_LRSCHEDULERS.register(
    name="warm_exp", interval="step", total_steps_key="total_steps"
)
class WarmupExpDecay(WarmupHoldScheduler):
    """Expontional decays lr to a min lr.

    ExpDecay curve is more gentle in the low lr range compared to Cosine scheduler, gives more compute
    budget for smaller learning rate. Recommand to combine with SGDs as they are usually set with larger
    lr than ADAMs, and we need stay more steps in small lrs.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        total_steps (int): Number of total training steps.
        warmup_steps (int, float): Number of warmup steps, when float means propotion. defaults to 0.
        pct_start (float, optional): Percent of steps before starting learning rate decay, defaults to 0.
        eta_min (int, optional): Minimum learning rate, defaults to 1e-6.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: Union[int, float] = 0,
        pct_start: float = 0.0,
        eta_min: float = 1e-6,
        last_epoch: int = -1,
        **kwargs,
    ):
        if not (0.0 <= pct_start <= 1.0):
            raise ValueError(f"pct_start must >= 0.0 and <= 1.0, got {pct_start}")

        if isinstance(warmup_steps, float):
            if not (0.0 <= warmup_steps <= 1.0):
                raise ValueError(
                    f"Passing float warmup_steps must >= 0.0 and <= 1.0, got {warmup_steps}"
                )
            warmup_steps = int(warmup_steps * total_steps)
        hold_steps = int(max((total_steps - warmup_steps), 0) * pct_start)
        anneal_steps = max(total_steps - warmup_steps - hold_steps, 0)
        initial_lrs = []
        for _, group in enumerate(optimizer.param_groups):
            if "initial_lr" in group:
                initial_lrs.append(group["initial_lr"])
            else:
                initial_lrs.append(group["lr"])

        exp_scale_fns = [
            partial(
                exp_scale_fn,
                initial_lr=initial_lr,
                total_steps=anneal_steps,
                eta_min=eta_min,
            )
            for initial_lr in initial_lrs
        ]
        base_scheduler = LambdaLR(optimizer, exp_scale_fns)
        super().__init__(
            optimizer,
            warmup_steps,
            hold_steps,
            base_scheduler=base_scheduler,
            last_epoch=last_epoch,
        )
