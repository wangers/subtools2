# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-09)

import warnings
from typing import Union

from .warmup import WARM_LRSCHEDULERS, WarmupHoldScheduler, _LRScheduler


class PolynomialLR(_LRScheduler):
    """Copied from pytorch, but add param: `end_lr` as min lr.

    Decays the learning rate of each parameter group using a polynomial function
    in the given total_iters. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_iters (int): The number of steps that the scheduler decays the learning rate. Default: 5.
        power (int): The power of the polynomial. Default: 1.0.
            power=0.5: squareroot, decays slow -> fast
            power=1: linear
            power=2: square, decays fast -> slow
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # xdoctest: +SKIP("undefined vars")
        >>> # Assuming optimizer uses lr = 0.001 for all groups
        >>> # lr = 0.001     if epoch == 0
        >>> # lr = 0.00075   if epoch == 1
        >>> # lr = 0.00050   if epoch == 2
        >>> # lr = 0.00025   if epoch == 3
        >>> # lr = 0.0       if epoch >= 4
        >>> scheduler = PolynomialLR(self.opt, total_iters=4, power=1.0)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
        self,
        optimizer,
        total_iters=5,
        power=1.0,
        end_lr: float = 0.0,
        last_epoch=-1,
        verbose=False,
    ):
        self.total_iters = total_iters
        self.power = power
        self.end_lr = end_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        decay_factor = (
            (1.0 - self.last_epoch / self.total_iters)
            / (1.0 - (self.last_epoch - 1) / self.total_iters)
        ) ** self.power
        return [
            max(group["lr"] * decay_factor, self.end_lr)
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        return [
            max(
                base_lr
                * (1.0 - min(self.total_iters, self.last_epoch) / self.total_iters)
                ** self.power,
                self.end_lr,
            )
            for base_lr in self.base_lrs
        ]


@WARM_LRSCHEDULERS.register(
    name="warm_poly", interval="step", total_steps_key="total_steps"
)
class WarmupPolynomial(WarmupHoldScheduler):
    """Polynomial learning rate scheduler with warmup.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        total_steps (int): Number of total training steps.
        warmup_steps (int, float): Number of warmup steps, when float means propotion. defaults to 0.
        pct_start (float, optional): Percent of steps before starting learning rate decay, defaults to 0.
        eta_min (int, optional): Minimum learning rate, defaults to 1e-6.
        power (float, optional): The power of polynomial, defaults to 1.0, which indicates linear.
            power=0.5: squareroot, decays slow -> fast
            power=1: linear
            power=2: square, decays fast -> slow
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: Union[int, float] = 0,
        pct_start: float = 0.0,
        eta_min: float = 0.0,
        power: float = 1.0,
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
        base_scheduler = PolynomialLR(
            optimizer, anneal_steps, end_lr=eta_min, power=power
        )
        super().__init__(
            optimizer,
            warmup_steps,
            hold_steps,
            base_scheduler=base_scheduler,
            last_epoch=last_epoch,
        )
