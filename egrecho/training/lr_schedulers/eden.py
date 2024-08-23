# Copyright      2022  Xiaomi Corp.        (authors: Daniel Povey)
#
# See ../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Copyright xmuspeech (Author: Leo 2024-04)
"""
PyTorch implementation of the Eden scheduler. Usually use with scaled adam.
Reference:
    Introduced in zipformer (ZIPFORMER: A FASTER AND BETTER ENCODER FOR AUTOMATIC SPEECH RECOGNITION).
    https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/optim.py
"""
import logging
from typing import List, Optional, Union

import torch
from torch.optim import Optimizer

from .warmup import WARM_LRSCHEDULERS, _LRScheduler


class LRScheduler(_LRScheduler):
    """
    Base-class for learning rate schedulers where the learning-rate depends on both the
    batch and the epoch. Unlike icefall, here it is a torch LRScheduler class version.

    NOTE:
        In pytorch LRScheduler, epoch and step only has the same notion: epoch. here we add a name `t_epoch` means
        truely epoch, and `last_epoch` in torch LRScheduler means steps, i.e., batch.
    """

    def __init__(
        self,
        optimizer,
        last_epoch=-1,
        verbose: bool = False,
    ):

        # epoch is step, and t_epoch is truelly epoch
        self.t_epoch = 0
        super().__init__(optimizer, last_epoch, verbose=verbose)

    def get_last_lr(self) -> List[float]:
        """Return last computed learning rate by current scheduler.  Will be a list of float."""
        return self._last_lr

    def get_lr(self):
        # Compute list of learning rates from self.epoch and self.batch and
        # self.base_lrs; this must be overloaded by the user.
        # e.g. return [some_formula(self.batch, self.epoch, base_lr) for base_lr in self.base_lrs ]
        raise NotImplementedError

    def step_batch(self, batch: Optional[int] = None) -> None:
        # Step the batch index, or just set it.  If `batch` is specified, it
        # must be the batch index from the start of training, i.e. summed over
        # all epochs.
        # You can call this in any order; if you don't provide 'batch', it should
        # of course be called once per batch.
        return self.step(batch)

    def step_epoch(self, epoch: Optional[int] = None):
        # Step the epoch index, or just set it.  If you provide the 'epoch' arg,
        # you should call this at the start of the epoch; if you don't provide the 'epoch'
        # arg, you should call it at the end of the epoch.
        if epoch is not None:
            self.t_epoch = epoch
        else:
            self.t_epoch = self.t_epoch + 1
        self._set_lrs()

    def _set_lrs(self):
        values = self.get_lr()
        assert len(values) == len(self.optimizer.param_groups)

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group["lr"] = lr
            self.print_lr(self.verbose, i, lr)
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def print_lr(self, is_verbose, group, lr, *args):
        """Display the current learning rate."""
        if is_verbose:
            print(
                f"Epoch={self.t_epoch}, batch={self.last_epoch}: adjusting learning rate"
                f" of group {group} to {lr:.4e}."
            )

    @property
    def batch(self):
        return self.last_epoch


class Eden(LRScheduler):
    """
    Eden scheduler.
    The basic formula (before warmup) is:
      lr = base_lr * (((batch**2 + lr_batches**2) / lr_batches**2) ** -0.25 *
                     (((epoch**2 + lr_epochs**2) / lr_epochs**2) ** -0.25)) * warmup
    where `warmup` increases from linearly 0.5 to 1 over `warmup_batches` batches
    and then stays constant at 1.

    If you don't have the concept of epochs, or one epoch takes a very long time,
    you can replace the notion of 'epoch' with some measure of the amount of data
    processed, e.g. hours of data or frames of data, with 'lr_epochs' being set to
    some measure representing "quite a lot of data": say, one fifth or one third
    of an entire training run, but it doesn't matter much.  You could also use
    Eden2 which has only the notion of batches.

    We suggest base_lr = 0.04 (passed to optimizer) if used with ScaledAdam

    Args:
        optimizer: the optimizer to change the learning rates on
        lr_batches: the number of batches after which we start significantly
              decreasing the learning rate, suggest 5000.
        lr_epochs: the number of epochs after which we start significantly
              decreasing the learning rate, suggest 6 if you plan to do e.g.
              20 to 40 epochs, but may need smaller number if dataset is huge
              and you will do few epochs.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr_batches: Union[int, float],
        lr_epochs: Union[int, float],
        warmup_steps: Union[int, float] = 500.0,
        warmup_start: float = 0.5,
        last_epoch: int = -1,
        verbose: bool = False,
    ):

        self.lr_batches = lr_batches
        self.lr_epochs = lr_epochs
        self.warmup_batches = warmup_steps

        assert 0.0 <= warmup_start <= 1.0, warmup_start
        self.warmup_start = warmup_start
        super(Eden, self).__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        factor = (
            (self.batch**2 + self.lr_batches**2) / self.lr_batches**2
        ) ** -0.25 * (
            ((self.t_epoch**2 + self.lr_epochs**2) / self.lr_epochs**2) ** -0.25
        )
        warmup_factor = (
            1.0
            if self.batch >= self.warmup_batches
            else self.warmup_start
            + (1.0 - self.warmup_start) * (self.batch / self.warmup_batches)
            # else 0.5 + 0.5 * (self.batch / self.warmup_batches)
        )

        return [x * factor * warmup_factor for x in self.base_lrs]


@WARM_LRSCHEDULERS.register(name="eden_s", interval="step")
class Eden2(LRScheduler):
    """
    Eden2 scheduler, simpler than Eden because it does not use the notion of epoch,
    only batches. Similar to `WarmupInvSqrt` actually.

    The basic formula (before warmup) is:
      lr = base_lr * ((batch**2 + lr_batches**2) / lr_batches**2) ** -0.5) * warmup

    where `warmup` increases from linearly 0.5 to 1 over `warmup_steps` batches
    and then stays constant at 1.


     E.g. suggest base_lr = 0.04 (passed to optimizer) if used with ScaledAdam

    Args:
        optimizer: the optimizer to change the learning rates on
        lr_batches: the number of batches after which we start significantly
              decreasing the learning rate, suggest 5000.
        warmup_steps (int, float): Number of warmup steps, when float means propotion.
        defaults to 0.

    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr_batches: Union[int, float],
        warmup_steps: Union[int, float] = 500.0,
        warmup_start: float = 0.5,
        last_epoch: int = -1,
        verbose: bool = False,
        **kwargs,
    ):
        self.lr_batches = lr_batches
        self.warmup_batches = warmup_steps

        assert 0.0 <= warmup_start <= 1.0, warmup_start
        self.warmup_start = warmup_start
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        factor = (
            (self.batch**2 + self.lr_batches**2) / self.lr_batches**2
        ) ** -0.5
        warmup_factor = (
            1.0
            if self.batch >= self.warmup_batches
            else self.warmup_start
            + (1.0 - self.warmup_start) * (self.batch / self.warmup_batches)
            # else 0.5 + 0.5 * (self.batch / self.warmup_batches)
        )

        return [x * factor * warmup_factor for x in self.base_lrs]


def _test_eden():
    from egrecho.training.optimizers.scaled_adam import ScaledAdam

    m = torch.nn.Linear(100, 100)
    optim = ScaledAdam(m.parameters(), lr=0.03)

    scheduler = Eden(optim, lr_batches=100, lr_epochs=2, verbose=True)

    for epoch in range(10):
        scheduler.step_epoch(epoch)  # sets epoch to `epoch`

        for step in range(20):
            x = torch.randn(200, 100).detach()
            x.requires_grad = True
            y = m(x)
            dy = torch.randn(200, 100).detach()
            f = (y * dy).sum()
            f.backward()

            optim.step()
            scheduler.step_batch()
            optim.zero_grad()

    logging.info(f"last lr = {scheduler.get_last_lr()}")
    logging.info(f"state dict = {scheduler.state_dict()}")


def _plot_eden_lr():
    import matplotlib.pyplot as plt

    from egrecho.training.optimizers.scaled_adam import ScaledAdam

    m = torch.nn.Linear(100, 100)

    for lr_epoch in [4, 10, 100]:
        for lr_batch in [100, 400]:
            optim = ScaledAdam(m.parameters(), lr=0.03)
            scheduler = Eden(
                optim, lr_batches=lr_batch, lr_epochs=lr_epoch, verbose=True
            )
            lr = []

            for epoch in range(10):
                scheduler.step_epoch(epoch)  # sets epoch to `epoch`

                for step in range(500):
                    lr.append(scheduler.get_lr())

                    x = torch.randn(200, 100).detach()
                    x.requires_grad = True
                    y = m(x)
                    dy = torch.randn(200, 100).detach()
                    f = (y * dy).sum()
                    f.backward()

                    optim.step()
                    scheduler.step_batch()
                    optim.zero_grad()
            plt.plot(lr, label=f"lr_epoch:{lr_epoch}, lr_batch:{lr_batch}")

    plt.legend()
    plt.savefig("lr.png")


## NOTE: FAILED run this file directly, please test in another test file.
# if __name__ == "__main__":
#     import logging
#     from egrecho.training.lr_schedulers.eden import _plot_eden_lr
#     import torch

#     torch.set_num_threads(1)
#     torch.set_num_interop_threads(1)
#     logging.getLogger().setLevel(logging.INFO)

#     _plot_eden_lr()
