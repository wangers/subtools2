# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-03)

import os
import random
import warnings
from contextlib import contextmanager
from random import getstate as python_get_rng_state
from random import setstate as python_set_rng_state
from typing import Any, Dict, Generator, Optional

import numpy as np
import torch

max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min


def set_all_seed(seed=42, include_cuda: bool = True):
    if not (min_seed_value <= seed <= max_seed_value):
        warnings.warn(
            f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}. "
            f"Trying to select a seed randomly."
        )
        seed = random.randint(min_seed_value, max_seed_value)
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if include_cuda:
        torch.cuda.manual_seed_all(seed)


def fix_cudnn(seed=42, deterministic=True, benchmark=False):
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark


class SeedWorkers:
    r"""
    Different workers with different seed.

    If provide ``rank``, it will randomlize acorssing node

    Args:
        seed (int)
            defaults to `42`.
        rank (int):
            defaults to `None`.
        include_cuda (bool):
            whether fix randomlize cuda. defaults to `False`.
    """

    def __init__(
        self, seed: int = 42, rank: Optional[int] = None, include_cuda: bool = False
    ):
        self.seed = seed
        self.rank = rank
        self.include_cuda = include_cuda

    def __call__(self, worker_id: int):
        seed = self.seed + worker_id
        if self.rank is not None:
            seed += 1000 * self.rank
        set_all_seed(seed, include_cuda=self.include_cuda)


@contextmanager
def isolate_rng(include_cuda: bool = True) -> Generator[None, None, None]:
    r"""
    A context manager that keeps track of the global random state, resets the global random state
    on exit to what it was before entering.

    It supports isolating the states for PyTorch, Numpy, and Python built-in random number generators.
    referring to:
    https://github.com/Lightning-AI/lightning/blob/master/src/lightning/pytorch/utilities/seed.py#isolate_rng

    Args:
        include_cuda: Whether to allow this function to also control the `torch.cuda` random number generator.
            Set this to ``False`` when using the function in a forked process where CUDA re-initialization is
            prohibited.

    Example:
        >>> import torch
        >>> torch.manual_seed(1)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> with isolate_rng():
        ...     [torch.rand(1) for _ in range(3)]
        [tensor([0.7576]), tensor([0.2793]), tensor([0.4031])]
        >>> torch.rand(1)
        tensor([0.7576])
    """
    states = _collect_rng_states(include_cuda)
    yield
    _set_rng_states(states)


def _collect_rng_states(include_cuda: bool = True) -> Dict[str, Any]:
    """Collect the global random state of :mod:`torch`, :mod:`torch.cuda`, :mod:`numpy` and Python."""
    states = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": python_get_rng_state(),
    }
    if include_cuda:
        states["torch.cuda"] = torch.cuda.get_rng_state_all()
    return states


def _set_rng_states(rng_state_dict: Dict[str, Any]) -> None:
    """Set the global random state of :mod:`torch`, :mod:`torch.cuda`, :mod:`numpy` and Python in the current
    process."""
    torch.set_rng_state(rng_state_dict["torch"])
    # torch.cuda rng_state is only included since v1.8.
    if "torch.cuda" in rng_state_dict:
        torch.cuda.set_rng_state_all(rng_state_dict["torch.cuda"])
    np.random.set_state(rng_state_dict["numpy"])
    version, state, gauss = rng_state_dict["python"]
    python_set_rng_state((version, tuple(state), gauss))
