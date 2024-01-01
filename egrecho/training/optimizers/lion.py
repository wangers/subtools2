# Copyright 2023 Google Research. All Rights Reserved.
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

# Copyright xmuspeech (Author: Leo 2023-12)

"""
PyTorch implementation of the Lion optimizer.
Reference:
    Symbolic Discovery of Optimization Algorithms.
    https://github.com/google/automl/blob/14d8bbd80417fd37711f3e9f9991431a77032f26/lion/lion_pytorch.py

"""

import math
from typing import Tuple

import torch
from torch.optim.optimizer import Optimizer

from .register import OPTIMIZERS_


@OPTIMIZERS_.register(name="lion")
class Lion(Optimizer):
    r"""Implements Lion algorithm."""

    def __init__(self, params, lr=1e-4, betas=(0.95, 0.98), weight_decay=0.1):
        """Initialize the hyperparameters.

        Experiments shows that a suitable learning rate for Lion is typically 3-10x smaller
        than that for AdamW. To maintain a similar strength, the value of weight_decay used for Lion is 3-10x
        larger than that for AdamW.

        Args:
          params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
          lr (float, optional): learning rate (default: 1e-4)
          betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.99) for vision (0.95, 0.98) for nlp.)
          weight_decay (float, optional): weight decay coefficient (default: 0.1)
        """

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
          closure (callable, optional): A closure that reevaluates the model
            and returns the loss.

        Returns:
          the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group["lr"] * group["weight_decay"])

                grad = p.grad
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # Weight update
                update = exp_avg.clone().mul_(beta1).add(grad * (1 - beta1))

                p.add_(update.sign_(), alpha=-group["lr"])

                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


def guess_lion_hparam(
    main_fan: int,
    alhpa: float = 1e-3,
    scale: bool = False,
) -> Tuple[float, float]:
    """
    Try to get lr and weight decay hparams.

    `lr = alhpa * std`, `weight_decay = alhpa / (2 * std)`

    Typically recommand:

        linear based: xiaver init case such as transformers, set main_fan with model dimention.
        conv based: kaimming init case such as resnet, set main_fan with channels * W * H,
        and the param scale should be True which will result in 2/fan_in.

    Reference:
        https://spaces.ac.cn/archives/9473

    Args:
        fan_dim: main fan of model (e.g., the dim of transformers.)
        alhpa: relative update of params in training begin. easy task: 1e-2, hard task: 1e-3
        fan_var: serves as our anticipation of the scale of parameter variations.
          For the multiplicative matrix can be directly initialized to the deviation, if
          not set, it will be `1/model_dim`.
    """
    std = math.sqrt(2 / main_fan) if scale else math.sqrt(1 / main_fan)

    return std * alhpa, alhpa / (2 * std)


@OPTIMIZERS_.register(name="rms_lion")
class RMSLion(Optimizer):
    """Lion algorithm but scaled lr accordding to rms, so we can set a larger lr.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.99) for classify.)
        weight_decay (float, optional): weight decay coefficient (default: 0.01)
    """

    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.99), weight_decay=0.01, min_param_rms=1e-5
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(
            lr=lr, betas=betas, weight_decay=weight_decay, min_param_rms=min_param_rms
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
          closure (callable, optional): A closure that reevaluates the model
            and returns the loss.

        Returns:
          the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.numel() > 1:
                    # param root mean square
                    param_rms = torch.sqrt(torch.mean(p**2)).clamp(
                        group["min_param_rms"]
                    )

                    lr = param_rms.detach() * group["lr"]
                else:
                    lr = group["lr"]

                # Perform stepweight decay
                p.data.mul_(1 - lr * group["weight_decay"])

                grad = p.grad
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)

                p.add_(update.sign_(), alpha=-lr)

                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss
