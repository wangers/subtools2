# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-09)

import inspect

import torch
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    OneCycleLR,
    ReduceLROnPlateau,
)
from typing_extensions import TypeAlias

from egrecho.utils.imports import _TORCH_GREATER_EQUAL_2_0
from egrecho.utils.register import Register, StrRegister

_TORCH_LRSCHEDULER: TypeAlias = (
    torch.optim.lr_scheduler.LRScheduler  # type: ignore[valid-type]
    if _TORCH_GREATER_EQUAL_2_0
    else torch.optim.lr_scheduler._LRScheduler
)


TORCH_LRSCHEDULERS = Register("scheduler")
LR_TOTAL_STEPS_KEY = StrRegister("total_steps")
_STEP_SCHEDULERS = (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    OneCycleLR,
)
schedulers = {
    k.lower(): v
    for k, v in lr_scheduler.__dict__.items()
    if not k.startswith("__")
    and k[0].isupper()
    and k != "_LRScheduler"
    and k != "LRScheduler"
    and inspect.isclass(v)
    and issubclass(v, _TORCH_LRSCHEDULER)
}
schedulers[ReduceLROnPlateau.__name__.lower()] = ReduceLROnPlateau
for name, sche_cls in schedulers.items():
    interval = "step" if issubclass(sche_cls, _STEP_SCHEDULERS) else "epoch"
    TORCH_LRSCHEDULERS.register(sche_cls, name=name, interval=interval)
LR_TOTAL_STEPS_KEY.register(["T_max", "total_iters"])
