# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Snowdar 2019-05-29)
#                     (Author: Leo 2023-07-29)

import math
import warnings
from typing import Dict

import torch
import torch.nn.functional as F


class MarginHead(torch.nn.Module):
    r"""This is a base class of margin heads. Besides apply margin penalty to its outputs,
    it will record the raw outputs (self.posterior) before penalty for computting a reliable accuracy.

    Examles:

        >>> def forward(self, inputs, targets):
        ...     outputs = self.linear(inputs)
        ...     self.posterior = outputs
        ...     outputs = balabala(outputs)
        ...     return outputs
    """

    def __init__(self):
        super().__init__()

        self._posterior = torch.empty(0)

    def forward(self, *inputs):
        raise NotImplementedError

    @property
    def posterior(self):
        return self._posterior

    @posterior.setter
    def posterior(self, rawout):
        self._posterior = rawout

    def get_init_margin(self):
        return getattr(self, "m", None)

    def update_margin(self, add_margin=None, lambda_m=1.0):
        """
        An inferce to contral margin dynamically during training.

        Args:
            add_margin (float, optional): Additional margin to be added. Default is None.
            lambda_m (float, optional): A smoothing factor for margin updates.
        """
        pass

    def from_name(
        name: str,
        input_dim: int,
        num_classes: int,
        sub_k: int = 1,
        do_topk: bool = False,
        **kwargs,
    ) -> "MarginHead":
        """
        Create an instance of MarginHead subclass based on its name.

        Args:
            name (str): The name of the margin head subclass.
            input_dim (int): The input dimension.
            num_classes (int): The number of labels.
            sub_k (int, optional): Sub-center parameter. Default is 1.
            do_topk (bool, optional): Flag to enable top-k penalty. Default is False.

        Returns:
            MarginHead: An instance of the specified MarginHead subclass.
        """
        if name == "linear":
            return LinearHead(input_dim, num_classes, sub_k=sub_k, **kwargs)
        elif name == "am":
            return AdditiveMarginHead(
                input_dim, num_classes, sub_k=sub_k, do_topk=do_topk, **kwargs
            )
        elif name == "aam":
            return ArcMarginHead(
                input_dim, num_classes, sub_k=sub_k, do_topk=do_topk, **kwargs
            )
        else:
            raise TypeError(
                f"Unsupported margin head type:{name}, choose from (linear, am, aam)"
            )


class LinearHead(MarginHead):
    """
    A linear projection with sub-center.

    Args:
        input_dim (int): The input dimension.
        num_classes (int): The number of labels.
        sub_k (int, optional): Sub-center parameter. Default is 1.
    """

    def __init__(self, input_dim, num_classes, sub_k=1, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.sub_k = max(1, sub_k)
        if kwargs.get("do_topk"):
            warnings.warn("Skip do_topk=True for linear head.")
        self.weight = torch.nn.Parameter(
            torch.randn(self.sub_k * num_classes, input_dim)
        )
        # torch.nn.init.xavier_normal_(self.weight, gain=1.0)
        torch.nn.init.normal_(self.weight, 0.0, 0.01)  # It seems better.

    def update_margin(self, add_margin=None, lambda_m=1.0):
        ...

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor = torch.empty(0)):
        """
        Forward pass of the linear head.

        Args:
            inputs (torch.Tensor): The input tensor.
            labels (torch.Tensor, optional): The target labels. Default is an empty tensor.

        Returns:
            torch.Tensor: The output tensor after applying the linear transformation.
        """
        cosine_theta = F.linear(F.normalize(inputs), F.normalize(self.weight))
        if self.sub_k > 1:
            cosine_theta = cosine_theta.reshape(-1, self.num_classes, self.sub_k)
            cosine_theta, _ = torch.max(cosine_theta, dim=2)
        self.posterior = cosine_theta.detach()  # For acc metrics.
        return cosine_theta

    def extra_repr(self):
        return f"(input_dim={self.input_dim}, num_classes={self.num_classes}, sub_k={self.sub_k} "


class AdditiveMarginHead(MarginHead):
    """
    Additive margin projection with sub-center and inter-topk penalty.

    Args:
        input_dim (int): The input dimension.
        num_classes (int): The number of classes.
        m (float, optional): The margin parameter. Default is 0.2.
        s (float, optional): The scale factor. Default is 30.0.
        sub_k (int, optional): Sub-center parameter. Default is 1.
        do_topk (bool, optional): Flag to enable top-k penalty. Default is False.
        topk_m (float, optional): The top-k margin parameter. Default is 0.06.
        topk (int, optional): The top-k parameter. Default is 5.

    Reference:
        - [1] Wang, F., Cheng, J., Liu, W., & Liu, H. (2018). Additive margin softmax for face verification.
        IEEE Signal Processing Letters, 25(7), 926-930.
        - [2] Miao Zhao, Yufeng Ma and Yiwei Ding et al., MULTI-QUERY MULTI-HEAD ATTENTION POOLING AND INTER-TOPK PENALTY FOR SPEAKER VERIFICATION.
    """

    def __init__(
        self,
        input_dim,
        num_classes,
        m=0.2,
        s=30.0,
        sub_k=1,
        do_topk=False,
        topk_m=0.06,
        topk=5,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.m = m  # margin
        self.s = s  # scale factor with feature normalization
        self.sub_k = max(1, sub_k)
        self.do_topk = do_topk
        self.topk = topk
        self.topk_m = topk_m
        self.topk_scale = topk_m / m
        self.weight = torch.nn.Parameter(
            torch.randn(self.sub_k * num_classes, input_dim)
        )
        # torch.nn.init.xavier_normal_(self.weight, gain=1.0)
        torch.nn.init.normal_(self.weight, 0.0, 0.01)  # It seems better.

        # margin states
        self.add_margin = m
        self.add_topk_margin = self.topk_scale * self.add_margin  # sync update
        self.lambda_m = 1  # smooth

    def update_margin(self, add_margin=None, lambda_m=1.0):
        """Update margin states."""
        self.lambda_m = lambda_m
        if add_margin is not None:
            self.add_margin = max(0, self.m + add_margin)
            self.add_topk_margin = self.topk_scale * self.add_margin

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor):
        """
        Forward pass of the additive margin head.

        Args:
            inputs (torch.Tensor): The input tensor.
            labels (torch.Tensor): The target labels.

        Returns:
            torch.Tensor: The output tensor after applying the additive margin transformation.
        """
        cosine_theta = F.linear(F.normalize(inputs), F.normalize(self.weight))
        if self.sub_k > 1:
            cosine_theta = cosine_theta.reshape(-1, self.num_classes, self.sub_k)
            cosine_theta, _ = torch.max(cosine_theta, dim=2)

        before_margin = self.s * cosine_theta
        self.posterior = before_margin.detach()  # For acc metrics.
        if not self.training:
            # For valid set.
            return self.posterior

        one_hot = F.one_hot(labels, self.num_classes)
        consine_theta_p = cosine_theta * one_hot

        cosine_theta_n = cosine_theta.masked_fill(
            one_hot.bool(), float("-inf")
        )  # filt out target.

        if self.do_topk and self.topk > 0:
            hard_threshold = torch.topk(cosine_theta_n, k=self.topk, dim=1)[0][
                :, -1:
            ].detach()  # (B, 1)
            hard_margin = torch.where(
                cosine_theta_n >= hard_threshold, self.add_topk_margin, 0.0
            ).detach()
        else:
            hard_margin = 0.0

        # penalty_cosine_theta [batch_size, num_class]
        # labels.unsqueeze(1) [batch_size, 1]
        penalty_cosine_theta = cosine_theta_n + hard_margin + self.add_margin

        penalty_cosine_theta.scatter_(
            1, labels.unsqueeze(1), 0.0
        )  # fill one hot position from -inf to zero.

        # n + p
        penalty_cosine_theta = penalty_cosine_theta + consine_theta_p

        penalty_cosine_theta = (
            self.lambda_m * penalty_cosine_theta + (1 - self.lambda_m) * cosine_theta
        )  # a warm version with propotion.
        penalty_cosine_theta *= self.s

        return penalty_cosine_theta

    def extra_repr(self):
        return (
            f"(input_dim={self.input_dim}, num_classes={self.num_classes}, sub_k={self.sub_k} "
            f"margin={self.m}, s={self.s}, do_topk={self.do_topk}, topk={self.topk}, topk_margin={self.topk_m}"
        )


class ArcMarginHead(MarginHead):
    """
    Arcface margin projection with sub-center and inter-topk penalty.

    Args:
        input_dim (int): The input dimension.
        num_classes (int): The number of classes.
        m (float, optional): The margin parameter. Default is 0.2.
        s (float, optional): The scale factor. Default is 32.0.
        sub_k (int, optional): Sub-center parameter. Default is 1.
        do_topk (bool, optional): Flag to enable top-k penalty. Default is False.
        topk_m (float, optional): The top-k margin parameter. Default is 0.06.
        topk (int, optional): The top-k parameter. Default is 5.
    """

    def __init__(
        self,
        input_dim,
        num_classes,
        m=0.2,
        s=32.0,
        sub_k=1,
        do_topk=False,
        topk_m=0.06,
        topk=5,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.m = m  # margin
        self.s = s  # scale factor with feature normalization
        self.sub_k = max(1, sub_k)
        self.do_topk = do_topk
        self.topk = topk
        self.topk_m = topk_m
        self.topk_scale = topk_m / m
        self.weight = torch.nn.Parameter(
            torch.randn(self.sub_k * num_classes, input_dim)
        )
        # torch.nn.init.xavier_normal_(self.weight, gain=1.0)
        torch.nn.init.normal_(self.weight, 0.0, 0.01)  # It seems better.

        # margin states
        self.add_margin = m
        self.add_topk_margin = self.topk_scale * self.add_margin  # sync update
        self.cos_m = math.cos(self.add_margin)
        self.sin_m = math.sin(self.add_margin)

        # to make arccos Monotonically decreasing lately
        self.th = math.cos(math.pi - self.add_margin)
        self.mm = math.sin(math.pi - self.add_margin)
        self.mmm = 1.0 + math.cos(math.pi - self.add_margin)

        self.lambda_m = 1  # smooth

    def update_margin(self, add_margin, lambda_m=1.0):
        """Update margin states."""
        self.lambda_m = lambda_m
        if add_margin is not None:
            self.add_margin = max(0, self.m + add_margin)
            self.add_topk_margin = self.topk_scale * self.add_margin
            self.cos_m = math.cos(self.add_margin)
            self.sin_m = math.sin(self.add_margin)

            self.th = math.cos(math.pi - self.add_margin)
            self.mm = math.sin(math.pi - self.add_margin)
            self.mmm = 1.0 + math.cos(math.pi - self.add_margin)

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor):
        """
        Forward pass of the arcface margin head.

        Args:
            inputs (torch.Tensor): The input tensor.
            labels (torch.Tensor): The target labels.

        Returns:
            torch.Tensor: The output tensor after applying the arcface margin transformation.
        """
        cosine_theta = F.linear(F.normalize(inputs), F.normalize(self.weight))
        if self.sub_k > 1:
            cosine_theta = cosine_theta.reshape(-1, self.num_classes, self.sub_k)
            cosine_theta, _ = torch.max(cosine_theta, dim=2)

        before_margin = self.s * cosine_theta
        self.posterior = before_margin.detach()  # For acc metrics.
        if not self.training:
            # For valid set.
            return self.posterior

        one_hot = F.one_hot(labels, self.num_classes)
        penalty_cosine_theta_p = (
            cosine_theta * self.cos_m
            - torch.sqrt(1.0 - torch.pow(cosine_theta, 2)) * self.sin_m
        )
        penalty_cosine_theta_p = (
            torch.where(
                cosine_theta > self.th,
                penalty_cosine_theta_p,
                cosine_theta - self.mmm,
            )
            * one_hot
        )

        if self.do_topk and self.topk > 0:
            cosine_theta_n = cosine_theta.masked_fill(
                one_hot.to(dtype=torch.bool), float("-inf")
            )  # filt out target.
            hard_threshold = torch.topk(cosine_theta_n, k=self.topk, dim=1)[0][
                :, -1:
            ].detach()  # (B, 1)
            hard_margin = torch.where(
                cosine_theta_n >= hard_threshold, self.add_topk_margin, 0.0
            ).detach()
            penalty_cosine_theta = torch.cos(torch.acos(cosine_theta) - hard_margin)
        else:
            penalty_cosine_theta = cosine_theta

        penalty_cosine_theta = (one_hot * penalty_cosine_theta_p) + (
            (1.0 - one_hot) * penalty_cosine_theta
        )
        penalty_cosine_theta = (
            self.lambda_m * penalty_cosine_theta + (1 - self.lambda_m) * cosine_theta
        )  # a warm version with propertion.
        penalty_cosine_theta *= self.s
        return penalty_cosine_theta

    def extra_repr(self):
        return (
            f"(input_dim={self.input_dim}, num_classes={self.num_classes}, sub_k={self.sub_k} "
            f"margin={self.m}, s={self.s}, do_topk={self.do_topk}, topk={self.topk}, topk_margin={self.topk_m}"
        )


#  Leo 2022-11-08
class MarginWarm:
    """
    Apply margin with warmup.

    Between start_epoch and end_epoch, the offset_margin is
    exponentially increasing from offset_margin (usually negative) to 0.
    The lambda_t is linearly increasing from init_lambda to 1.
    It is designed to control the MarginHead through `margin + offset_margin` and
    `penalty_cosine_theta = lambda * penalty_cosine_theta + (1 - lambda) * cosine_theta`

    Args:
        head (Module): head moudle to apply margin.
        start_epoch (int): The epoch when the margin warmup starts.
        end_epoch (int): The epoch when the margin warmup ends.
        offset_margin (float, optional): The initial offset margin value.
            scaled from a negative value to 0.0 during training.
            If negative: scaled from it.
            If None: it will infered by head module's attr `init_margin`.
            If set 0.0: close it.
            Default: None
        init_lambda (float, optional): The initial value of lambda (default is 1.0).
            Set to a value between (0, 1) to open it.
        num_steps_per_epoch (int, optional): The number of iterations per epoch (default is None).
        enable_resume (bool): Control the :method:`self.resume` to set `last_epoch` (default is True).
    """

    def __init__(
        self,
        head: MarginHead,
        start_epoch,
        end_epoch,
        offset_margin=None,
        init_lambda=1.0,
        num_steps_per_epoch=None,
        last_epoch=-1,
        enable_resume=True,
    ):
        super().__init__()
        assert start_epoch >= 0
        if end_epoch < start_epoch:
            raise ValueError(
                "End_epoch should not smaller then start_epoch, but got end_epoch: {}, start_epoch:{}".format(
                    end_epoch, start_epoch
                )
            )
        if not abs(init_lambda - 0.5 <= 0.5):
            raise ValueError(
                f"init_lambda should be in [0, 1],but got ({init_lambda})."
            )
        self.head = head
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        if offset_margin is None:
            offset_margin = -(self.head.get_init_margin() or 0.0)
        self.offset_margin = offset_margin
        self.init_lambda = init_lambda
        self.num_steps_per_epoch = num_steps_per_epoch
        self.last_epoch = last_epoch
        if num_steps_per_epoch:
            self.update_step_range(num_steps_per_epoch)

        self.offset_stat = self.offset_margin
        self.lambda_t_stat = init_lambda
        self.enable_resume = enable_resume

    def update_step_range(self, num_steps_per_epoch, overwrite=True):
        """
        Update the step range (iter number) for margin warmup.

        Args:
            num_steps_per_epoch (int): The number of iterations per epoch.
            overwrite (bool, optional): Whether to overwrite the existing epoch_iter value (default is True).
        """
        if not overwrite and self.num_steps_per_epoch:
            raise ValueError(
                "num_steps_per_epoch has been set as {}, but overwrite = {} now".format(
                    self.num_steps_per_epoch, overwrite
                )
            )
        else:
            self.num_steps_per_epoch = num_steps_per_epoch
        self.increase_start_iter = self.start_epoch * num_steps_per_epoch
        self.fix_start_iter = self.end_epoch * num_steps_per_epoch
        self.step_range = self.fix_start_iter - self.increase_start_iter

    def get_increase_margin(self, cur_step):
        """
        Calculate the offset margin and lambda_t during the warmup period.

        Args:
            cur_step (int): The current step.

        Returns:
            Tuple[float, float]: The offset margin and lambda_t.
        """
        initial_val = 1.0
        final_val = 1e-3

        cur_pos = cur_step - self.increase_start_iter

        ratio = (
            math.exp(
                (cur_pos / self.step_range) * math.log(final_val / (initial_val + 1e-6))
            )
            * initial_val
        )
        offset_margin = self.offset_margin * ratio

        lambda_t = self.init_lambda + (cur_pos / self.step_range) * (
            1 - self.init_lambda
        )

        return offset_margin, lambda_t

    def step(self, cur_step=None):
        """
        Perform a step in margin warmup.

        Args:
            cur_step (int): The current step.

        Returns:
            Tuple[float, float]: The offset margin and lambda_t for the current step.
        """
        if self.num_steps_per_epoch < 0 or not isinstance(
            self.num_steps_per_epoch, int
        ):
            raise ValueError(
                "MarginWarm expected positive integer num_steps_per_epoch, but got {}".format(
                    self.num_steps_per_epoch
                )
            )
        if cur_step is None:
            self.last_epoch += 1
            cur_step = self.last_epoch
        else:
            self.last_epoch = cur_step

        if cur_step >= self.fix_start_iter:
            offset_margin, lambda_t = 0.0, 1.0

        elif cur_step > self.increase_start_iter:
            offset_margin, lambda_t = self.get_increase_margin(cur_step)

        else:
            offset_margin, lambda_t = self.offset_margin, self.init_lambda

        self.offset_stat, self.lambda_t_stat = offset_margin, lambda_t
        self.head.update_margin(offset_margin, lambda_t)

    def get_stats(self) -> Dict[str, float]:
        stats = {}
        if self.init_lambda < 1:
            stats["margin_lambda"] = self.lambda_t_stat
        if self.offset_margin < 0.0:
            stats["margin_offset"] = self.offset_stat
        return stats

    def set_step(self, last_epoch):
        self.last_epoch = last_epoch

    def resume(self, last_epoch):
        if self.enable_resume:
            self.set_step(last_epoch)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(head={self.head.__class__.__name__}, "
            f"start_epoch={self.start_epoch}, end_epoch={self.end_epoch}, init_offset_margin={self.offset_margin}, "
            f"num_steps_per_epoch={self.num_steps_per_epoch}, init_lambda={self.init_lambda})"
        )
