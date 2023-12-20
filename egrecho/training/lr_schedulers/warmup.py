# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-09)

from typing import Optional

from egrecho.utils.imports import _TORCH_GREATER_EQUAL_2_0
from egrecho.utils.register import Register

if _TORCH_GREATER_EQUAL_2_0:
    from torch.optim.lr_scheduler import LRScheduler as _LRScheduler
else:
    from torch.optim.lr_scheduler import _LRScheduler

WARM_LRSCHEDULERS = Register("warm_scheduler")


class WarmupHoldScheduler(_LRScheduler):
    """Starts with a linear warmup lr schedule until it reaches N steps and a flat lr schedule
    until it reaches M steps then applies the specific scheduler.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        warmup_steps (int): Number of steps to linearly warmup lr until starting applying the scheduler.
        hold_steps (int): Number of steps to keep the initial lr until starting applying the scheduler.
        base_scheduler (:class:`torch.optim.lr_scheduler`): After target_epoch, use this scheduler.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int = 0,
        hold_steps: int = 0,
        base_scheduler: Optional[_LRScheduler] = None,
        last_epoch: int = -1,
    ):
        if optimizer != base_scheduler.optimizer:
            raise ValueError(
                "WarmupHoldScheduler expects provide same optimizer in warmup stage and post scheduler, but "
                "got different optimizers."
            )
        if hold_steps < 0:
            raise ValueError(f"hold_steps must >= 0, got {hold_steps}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must >= 0, got {warmup_steps}")
        self.warmup_steps = warmup_steps
        self.hold_steps = hold_steps
        self._base_scheduler = base_scheduler
        self.warmuphold_done = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self._is_lr_warmuphold():
            return self._warmuphold_lr(self.last_epoch)
        else:
            if not self.warmuphold_done:
                self._base_scheduler.base_lrs = self.base_lrs
                # reset lr to base_lr
                for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                    group["lr"] = base_lr
                self.warmuphold_done = True
            with _enable_get_lr_call(self._base_scheduler):
                return self._base_scheduler.get_lr()

    def _is_lr_warmuphold(self):
        """
        Check if we're in warmup and hold stage.
        """
        return self.last_epoch < self.warmup_steps + self.hold_steps

    def _warmuphold_lr(self, step):
        """
        Return lr in warmup and hold stage.
        """
        if step < self.warmup_steps:
            return [(step + 1) / self.warmup_steps * lr for lr in self.base_lrs]
        if step < self.warmup_steps + self.hold_steps:
            return self.base_lrs

    def step(self, epoch=None):
        if self.warmuphold_done:
            if epoch is None:
                self.last_epoch += 1
                self._base_scheduler.step()
            else:
                self.last_epoch = epoch
                self._base_scheduler.step(epoch - (self.warmup_steps + self.hold_steps))
            self._last_lr = self._base_scheduler.get_last_lr()
        else:
            return super().step(epoch)

    def state_dict(self):
        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "_base_scheduler")
        }
        if self._base_scheduler and isinstance(self._base_scheduler, _LRScheduler):
            state_dict["_base_scheduler"] = self._base_scheduler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        base_scheduler = state_dict.pop("_base_scheduler", None)
        self.__dict__.update(state_dict)

        if base_scheduler:
            # restore state dict if popped.
            state_dict["_base_scheduler"] = base_scheduler
            self._base_scheduler.load_state_dict(base_scheduler)


class _enable_get_lr_call:
    def __init__(self, o):
        self.o = o

    def __enter__(self):
        self.o._get_lr_called_within_step = True
        return self

    def __exit__(self, type, value, traceback):
        self.o._get_lr_called_within_step = False
