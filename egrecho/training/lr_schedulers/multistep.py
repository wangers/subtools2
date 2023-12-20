# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-09)

from typing import List, Optional

from torch.optim.lr_scheduler import MultiStepLR

from .warmup import WARM_LRSCHEDULERS, WarmupHoldScheduler


@WARM_LRSCHEDULERS.register(name="warm_multistep", interval="step")
class WarmupMultistep(WarmupHoldScheduler):
    """
    Decays the learning rate of each parameter group by gamma once the
    number of epoch reaches one of the milestones. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside
    this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of warmup steps.
        milestones (list): List of epoch indices. If not set, no decays applied.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int = 0,
        milestones: Optional[List[int]] = None,
        gamma: float = 0.1,
        last_epoch: int = -1,
        **kwargs,
    ):
        milestones = milestones or []
        milestones = [v - warmup_steps for v in milestones if v >= warmup_steps]
        base_scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        super().__init__(
            optimizer,
            warmup_steps,
            base_scheduler=base_scheduler,
            last_epoch=last_epoch,
        )


@WARM_LRSCHEDULERS.register(name="warm_stay", interval="step")
class WarmupStay(WarmupMultistep):
    """
    Warmup then hold on.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of warmup steps.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int = 0,
        last_epoch: int = -1,
        **kwargs,
    ):
        super().__init__(
            optimizer,
            warmup_steps,
            last_epoch=last_epoch,
        )
