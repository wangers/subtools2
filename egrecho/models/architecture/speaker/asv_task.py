# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-08)

from copy import deepcopy
from typing import Any, Callable, Dict, Optional

import torch.nn.functional as F
from torchmetrics import Accuracy

from egrecho.core.teacher import LR_SCHEDULER_TYPE, OPTIMIZER_TYPE, ModuleDict, Teacher
from egrecho.nn.head import LinearHead, MarginHead, MarginWarm
from egrecho.utils.logging import get_logger
from egrecho.utils.misc import ConfigurationException
from egrecho.utils.torch_utils import tensor_has_nan

logger = get_logger()


class ClassificationMixin:
    """
    Add metrics and ce loss.
    """

    def _build(
        self,
        num_classes: Optional[int] = None,
        loss_fn: Optional[Callable] = None,
        metrics: Optional[Any] = None,
    ):
        self.num_classes = num_classes

        if metrics is None:
            metrics = Accuracy(task="multiclass", num_classes=self.num_classes)

        if loss_fn is None:
            loss_fn = F.cross_entropy

        return metrics, loss_fn


class SVTeacher(ClassificationMixin, Teacher):
    r"""A teacher to train speaker classification task.

    Args:
        num_classes (int):
            The number of labels in the classification task.
        margin_warm (bool):
            Whether to use margin scheduler for learning rate.
        margin_warm_kwargs (dict):
            Keyword arguments for margin warm-up scheduler. These arguments are used to configure the
            margin warm-up behavior. If not set (None), default values will be used. Default values are
            `{"start_epoch": 7.5, "end_epoch": 15, "offset_margin": -0.2}`.
        optimizer (OPTIMIZER_TYPE):
            The optimizer to use for training. It can be a string representing the optimizer's name, or a
            tuple like `('adam', {"weight_decay": 0.01})` where the first element is the optimizer's name, and
            the second element is a dictionary of optimizer-specific configuration. If not set (empty string),
            the default value `('adamw', {"weight_decay": 0.1})` will be used.
        lr_scheduler (LR_SCHEDULER_TYPE):
            The learning rate (LR) scheduler to use during training. It can be a string representing the LR
            scheduler's name, or a tuple with optional dictionaries for configuration. You can choose from
            available LR schedulers using `self.available_lr_schedulers()`. Examples include:
        - A single string: e.g., 'warm_cosine'.
        - A tuple with a dictionary: e.g., `('warm_cosine', {"warmup_steps": 20000})`.
        - A tuple with two dictionaries:
        e.g., `('warm_cosine', {"warmup_steps": 20000}, {'interval': "step"})`. while the last dict is used to
        overwrite lr config of `egrecho.core.teacher.DEFAULT_PL_LRCONFIG`
        which is used to compose `lightning` lr scheduler.
    """

    task_name = "automatic-speaker-verification"

    def __init__(
        self,
        num_classes: int = None,
        margin_warm: bool = True,
        margin_warm_kwargs: dict = None,
        optimizer: OPTIMIZER_TYPE = "",
        lr_scheduler: LR_SCHEDULER_TYPE = "",
        lr: Optional[float] = 0.01,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.margin_warm = margin_warm
        self.margin_warm_kwargs = margin_warm_kwargs or {
            "start_epoch": 7.5,
            "end_epoch": 15,
        }
        self.optimizer = optimizer or ("adamw", {"weight_decay": 0.1})
        self.lr_scheduler = lr_scheduler or ("warm_cosine", {"warmup_steps": 20000})
        self.lr = lr

        self.margin_scheduler = None

    def setup_model(self):
        acc_metric, loss_fn = self._build(self.num_classes)

        self.setup_loss_fn_dict({"loss": loss_fn})
        metrics = {"acc": acc_metric}
        self.setup_train_metrics(metrics)
        self.setup_val_metrics(deepcopy(metrics))
        if (
            getattr(self.model, "head", None)
            and isinstance(self.model.head, MarginHead)
            and not isinstance(self.model.head, LinearHead)
            and self.margin_warm
        ):
            self.margin_scheduler = MarginWarm(
                self.model.head, **self.margin_warm_kwargs
            )

    def common_step(self, batch: dict, batch_idx: int, metrics: ModuleDict) -> Dict:
        if isinstance(batch, dict):
            labels = batch.pop("labels")
            embd = self.model(**batch)
        else:
            x, labels = batch
            embd = self.model(x)
        if isinstance(embd, (list, tuple)):
            embd = embd[0]
        out_hat = self.model.head(embd, labels)

        posterior = (
            self.model.head.posterior
            if hasattr(self.model.head, "posterior")
            else out_hat
        )
        outputs = {"output": out_hat}
        losses = {
            name: l_fn(out_hat, labels) for name, l_fn in self.loss_fn_dict.items()
        }
        outputs["loss"] = losses["loss"]
        logs = {}
        metrics["acc"](posterior, labels)
        logs["acc"] = metrics["acc"]  # add metrics to log.

        logs.update(losses)  # add loss to log.
        outputs["logs"] = logs

        return outputs

    def training_step(self, batch, batch_idx):
        if self.margin_scheduler:
            self.margin_scheduler.step()

        outputs = self.common_step(batch, batch_idx, self.train_metrics)

        if tensor_has_nan(outputs["loss"]):
            # skip batch if something went wrong for some reason
            logger.warning(f"Got NaN loss in batch_idx:{batch_idx}, skip it.")

            return None

        self.model.log_dict(
            {f"train_{k}": v for k, v in outputs["logs"].items()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        if self.margin_scheduler:
            margin_stats = self.margin_scheduler.get_stats()
            if margin_stats:
                self.model.log_dict(margin_stats, prog_bar=True, on_epoch=False)
        return outputs["loss"]

    def validation_step(self, batch, batch_idx):
        outputs = self.common_step(batch, batch_idx, self.val_metrics)
        if tensor_has_nan(outputs["loss"]):
            # skip batch if something went wrong for some reason
            logger.warning(f"Got NaN loss in batch_idx:{batch_idx}, skip it.")
            return None
        self.model.log_dict(
            {f"val_{k}": v for k, v in outputs["logs"].items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return outputs["loss"]

    def on_train_start(self):
        if self.margin_scheduler:
            self._update_marginwarm()

            if self.trainer.global_step > 0:
                self.margin_scheduler.resume(self.trainer.global_step)
                self.model.print(
                    f"#### Resume a marginwarm as {self.margin_scheduler}."
                )

    def _update_marginwarm(self):
        if self.margin_scheduler and self.margin_scheduler.num_steps_per_epoch is None:
            estimate_steps_per_epoch = self.estimated_num_steps_per_epoch
            if estimate_steps_per_epoch in (float("inf"), None):
                raise ConfigurationException(
                    f"{self.__class__.__name__} failed detect estimate_steps_per_epoch, "
                    f"got {estimate_steps_per_epoch}. try configure `num_steps_per_epoch` in `margin_warm_kwargs` "
                    "directly or configure a trainer can infer batches in one epoch."
                )
            self.margin_scheduler.update_step_range(estimate_steps_per_epoch)
            self.model.print(
                f"#### Infered {estimate_steps_per_epoch} steps per epoch for marginwarm, it may not absolutely accurate."
            )

    def configure_optimizers(self):
        return self.configure_single_optimizer(
            self.optimizer, self.lr_scheduler, self.lr
        )
