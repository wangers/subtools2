# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-09)

import math
from functools import partial
from typing import Optional

from torch.optim.lr_scheduler import LambdaLR

from .warmup import WARM_LRSCHEDULERS, WarmupHoldScheduler


def invsqrt_scale_fn(current_step: int, invsqrt_gamma: int):
    invsqrt_gamma = max(1, invsqrt_gamma)
    decay = math.sqrt(invsqrt_gamma / (invsqrt_gamma + current_step))
    return decay


@WARM_LRSCHEDULERS.register(name="warm_invsqrt", interval="step")
class WarmupInvSqrt(WarmupHoldScheduler):
    """Decays at an inverse square root rate. Can be used without total step.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        total_steps (int): Number of total training steps.
        warmup_steps (int, float): Number of warmup steps, when float means propotion. defaults to 0.
        hold_steps (int): Number of steps to keep the initial lr until starting applying the scheduler.
        invsqrt_gamma (int, optional): Determines the cycle length of the inverse square root,
        if not set, it is inferred by warmup_steps adds hold steps.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int = 0,
        hold_steps: int = 0,
        invsqrt_gamma: Optional[int] = None,
        last_epoch: int = -1,
        **kwargs,
    ):
        if isinstance(warmup_steps, int) ^ isinstance(hold_steps, int):
            raise ValueError(
                f"warmup_steps and hold_steps must be int, got warmup_steps: {type(warmup_steps):{warmup_steps}}, "
                f"hold_steps: {type(hold_steps):{hold_steps}}."
            )
        warmup_steps = max(0, warmup_steps)
        hold_steps = max(0, hold_steps)
        if not invsqrt_gamma:
            invsqrt_gamma = warmup_steps + hold_steps
            if not invsqrt_gamma:
                raise ValueError(
                    "If not provide invsqrt_gamma, defaulting to set to "
                    f"`warmup_steps+hold_steps`, but got a  valid sum number {invsqrt_gamma}."
                )

        invsqrt_fn = partial(invsqrt_scale_fn, invsqrt_gamma=invsqrt_gamma)

        base_scheduler = LambdaLR(optimizer, invsqrt_fn)
        super().__init__(
            optimizer,
            warmup_steps,
            hold_steps,
            base_scheduler=base_scheduler,
            last_epoch=last_epoch,
        )
