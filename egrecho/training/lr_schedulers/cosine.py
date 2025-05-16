# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-09)

from typing import Union

from torch.optim.lr_scheduler import CosineAnnealingLR

from .warmup import WARM_LRSCHEDULERS, WarmupHoldScheduler


@WARM_LRSCHEDULERS.register(
    name="warm_cosine", interval="step", total_steps_key="total_steps"
)
class WarmupHoldCosineLR(WarmupHoldScheduler):
    """
    Cosine annealing learning rate scheduler with learning rate warmup. A linear warmup schedule will be
    applied, and then the learning rate will be a fixed value before starting decay.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        total_steps (int): Total number of training steps.
        warmup_steps (int, float): Number of warmup steps, when float means propotion. defaults to 0.
        pct_start (int, float, optional): Percent of steps before starting learning rate decay, defaults to 0.
        eta_min (int, optional): Minimum learning rate, defaults to 1e-6.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: Union[int, float] = 0,
        pct_start: Union[int, float] = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
        **kwargs,
    ):

        if isinstance(warmup_steps, float):
            if not (0.0 <= warmup_steps <= 1.0):
                raise ValueError(
                    f"Passing float warmup_steps must >= 0.0 and <= 1.0, got {warmup_steps}"
                )
            warmup_steps = int(warmup_steps * total_steps)

        if isinstance(pct_start, float):
            if not (0.0 <= pct_start <= 1.0):
                raise ValueError(
                    f"float pct_start must >= 0.0 and <= 1.0, got {pct_start}"
                )
            hold_steps = int(max((total_steps - warmup_steps), 0) * pct_start)
        elif isinstance(pct_start, int):
            if not (pct_start >= 0):
                raise ValueError(f"int pct_start must >= 0, got {pct_start}")
            hold_steps = pct_start
        else:
            raise TypeError(f'{type(hold_steps)}')

        anneal_steps = max(total_steps - warmup_steps - hold_steps, 0)
        base_scheduler = CosineAnnealingLR(optimizer, anneal_steps, eta_min=eta_min)
        super().__init__(
            optimizer,
            warmup_steps,
            hold_steps,
            base_scheduler,
            last_epoch=last_epoch,
        )
