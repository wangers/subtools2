# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-08)


import collections
import copy
import inspect
import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torchmetrics
from torch.nn import Module, ModuleDict
from torch.optim.optimizer import Optimizer
from torchmetrics import Metric

from egrecho.core.model_out import ModelOutput
from egrecho.core.optimization import (
    _TORCH_LRSCHEDULER,
    LR_TOTAL_STEPS_KEY,
    OPTIMIZERS_,
    TORCH_LRSCHEDULERS,
    WARM_LRSCHEDULERS,
)
from egrecho.utils.register import Register, StrRegister

METRICS_TYPE = Union[Metric, Sequence[Metric], Mapping[str, Metric], None]
LOSS_FN_TYPE = Union[Callable, Mapping, Sequence, None]
OPTIMIZER_TYPE = Union[str, Callable, Tuple[str, Dict[str, Any]], None]
LR_SCHEDULER_TYPE = Union[
    str,
    Callable,
    Tuple[str, Dict[str, Any]],
    Tuple[str, Dict[str, Any], Dict[str, Any]],
    None,
]
DEFAULT_PL_LRCONFIG = {
    "scheduler": None,
    "name": None,
    "interval": "epoch",
    "frequency": 1,
    "reduce_on_plateau": False,
    "monitor": None,
    "strict": True,
    "opt_idx": None,
}
LRSCHEDULERS = Register("schedulers")
LRSCHEDULERS += TORCH_LRSCHEDULERS
LRSCHEDULERS += WARM_LRSCHEDULERS


def _no_ops(x):
    return x


class DefaultOut(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


class Teacher:
    """
    Base class for teacher aims to provide an access to data, criterion for model, and step action details.

    Usually, the step details in fit (train + validation) stage is related to the data and objectives (criterion),
    combines with model's `method::forward`. A teacher inheriets from `Lightning-AI`'s `LightningDataModule`
    extracts the logics of traning step, make data and objective independent with model forward.
    In addition to `LightningDataModule`, it adds a point to model to do stepping. key methods:
        - :method:``train_dataloader`` & :method:``val_dataloader``: access to data.
        - :method:``training_step`` & :method:``validation_step``: detail step logics.
        - :method:``setup_model``:
            - called in :method:`self.model.setup`.
            - build models dynamically or adjust something about them at the beginning of fit stage.
            see :method:``pl.LightningDataModul.setup``

    In this class, :method:``training_step`` & :method:``validation_step``, :method:``setup_model` are simple
    examples referring to:
        https://github.com/Lightning-Universe/lightning-flash/blob/master/src/flash/core/model.py#Task
    Feel free to modify them in derived implementations.

    Class attributes (overridden by derived classes):
        - **task_name** (`str`) -- name of task, e.g., `"automatic-speaker-verification"`.
    """

    optimizers_registry: Register = OPTIMIZERS_
    lr_schedulers_registry: Register = LRSCHEDULERS
    lr_total_steps_key_registry: StrRegister = LR_TOTAL_STEPS_KEY
    task_name = None

    def __init__(
        self,
        loss_fn: LOSS_FN_TYPE = None,
        metrics: METRICS_TYPE = None,
        inputs_transform: Optional[Callable] = _no_ops,
        **kwargs,
    ) -> None:
        super().__init__()
        from lightning.pytorch import LightningModule

        # Pointer to the model object.
        self._model: LightningModule = None
        self._loss_fn = {} if loss_fn is None else normalize_callable_dict(loss_fn)
        self._metrics = {} if metrics is None else normalize_callable_dict(metrics)
        self.inputs_transform = inputs_transform

    @property
    def model(self):
        return self._model

    @property
    def trainer(self):
        if not getattr(self, "model", None):
            raise RuntimeError("The Teacher isn't attached to the model yet.")

        if not getattr(self.model, "trainer", None):
            raise RuntimeError("The Teacher isn't attached to the trainer yet.")
        return self.model.trainer

    @property
    def train_metrics(self):
        return getattr(self.model, "train_metrics", {})

    @property
    def val_metrics(self):
        return getattr(self.model, "val_metrics", {})

    def attach_model(self, model: Module):
        """Attach model, do this once before step."""
        self._model = model

    def setup(self):
        """
        Called by linked model's hook: :method:`setup` in fit stage, gives a chance to setup in accelerate environment.
        """

    def setup_model(self):
        """
        An inferce for building linked model.
        """
        self.setup_loss_fn()
        self.setup_metrics()

    def setup_loss_fn(self):
        self.model.loss_fn = self._loss_fn

    def setup_metrics(self):
        self.model.train_metrics = ModuleDict(self._metrics)
        self.model.val_metrics = ModuleDict(copy.deepcopy(self._metrics))

    def _prepare_batch(self, batch) -> dict:
        return self.inputs_transform(batch)

    def common_step(
        self, batch: dict, batch_idx: int, metrics: ModuleDict, prefix: str
    ) -> Dict:
        batch = self._prepare_batch(batch)
        labels = batch.pop("labels")
        out_hat = self.model(**batch)
        outputs = {"output": out_hat}
        out_hat = self.to_loss_format(out_hat)
        losses = {
            name: l_fn(out_hat, labels) for name, l_fn in self.model.loss_fn.items()
        }
        logs = {}
        out_hat = self.to_metrics_format(out_hat)

        for name, metric in metrics.items():
            if isinstance(metric, torchmetrics.metric.Metric):
                metric(out_hat, labels)
                logs[name] = metric  # torchmetrics needs two line
            else:
                logs[name] = metric(out_hat, labels)

        if len(losses.values()) > 1:
            logs["total_loss"] = sum(losses.values())
            return logs["total_loss"], logs
        outputs["loss"] = self.compute_loss(losses)
        outputs["logs"] = self.compute_logs(logs, losses)  # add metrics to log.
        return outputs

    def compute_metrics(self, *args, **kwargs):
        ...

    def compute_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        return list(losses.values())[0]

    def compute_logs(self, logs: Dict[str, Any], losses: Dict[str, torch.Tensor]):
        logs.update(losses)
        return logs

    @staticmethod
    def to_loss_format(x: torch.Tensor) -> torch.Tensor:
        """
        Some case we need transform loss.
        """
        return _no_ops(x)

    @staticmethod
    def to_metrics_format(x: torch.Tensor) -> torch.Tensor:
        """
        Some case we need transform loss.
        """
        return _no_ops(x)

    def training_step(self, batch, batch_idx):
        outputs = self.common_step(batch, batch_idx, self.train_metrics, "train")
        if hasattr(self.model, "log_dict") and isinstance(
            self.model.log_dict, collections.abc.Callable
        ):
            self.model.log_dict(
                {f"train_{k}": v for k, v in outputs["logs"].items()},
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
        return outputs["loss"]

    def validation_step(self, batch, batch_idx):
        outputs = self.common_step(batch, batch_idx, self.val_metrics, "val")
        if hasattr(self.model, "log_dict") and isinstance(
            self.model.log_dict, collections.abc.Callable
        ):
            self.model.log_dict(
                {f"val_{k}": v for k, v in outputs["logs"].items()},
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
        return outputs["loss"]

    def on_fit_start(self) -> None:
        """Called at the very beginning of fit.

        If on DDP it is called on every process

        """

    def on_fit_end(self) -> None:
        """Called at the very end of fit.

        If on DDP it is called on every process

        """

    @classmethod
    def available_optimizers(cls) -> List[str]:
        """Returns a list containing the keys of the available Optimizers."""
        registry: Optional[Register] = getattr(cls, "optimizers_registry", None)
        if registry is None:
            return []
        return registry.keys()

    @classmethod
    def available_lr_schedulers(cls) -> List[str]:
        """Returns a list containing the keys of the available LR schedulers."""
        registry: Optional[Register] = getattr(cls, "lr_schedulers_registry", None)
        if registry is None:
            return []
        return registry.keys()

    @classmethod
    def available_lr_scheduler_total_steps_name(cls) -> List[Tuple[str, str]]:
        lr_scheduler_keys = cls.available_lr_schedulers()
        rs = []

        for lr_key in lr_scheduler_keys:
            total_steps_name = cls.get_lr_scheduler_total_steps_name(lr_key)
            if total_steps_name:
                rs.append((lr_key, total_steps_name))
        return rs

    @classmethod
    def _get_optimizer_class_from_registry(cls, optimizer_key: str) -> Optimizer:
        if optimizer_key.lower() not in cls.available_optimizers():
            raise KeyError(
                f"Please provide a valid optimizer name and make sure it is registerd with the Optimizer registry."
                f"\nUse `{cls.__name__}.available_optimizers()` to list the available optimizers."
                f"\nList of available Optimizers: {cls.available_optimizers()}."
            )
        return cls.optimizers_registry.get(optimizer_key.lower())

    @classmethod
    def _get_lr_scheduler_class_from_registry(
        cls, lr_scheduler_key: str
    ) -> Dict[str, Any]:
        if lr_scheduler_key.lower() not in cls.available_lr_schedulers():
            raise KeyError(
                f"Please provide a valid scheduler name and make sure it is registerd with the Scheduler registry."
                f"\nUse `{cls.__name__}.available_lr_schedulers()` to list the available schedulers."
                f"\n>>> List of available LR Schedulers: {cls.available_lr_schedulers()}."
            )
        lr_scheduler_fn: Dict[str, Any] = cls.lr_schedulers_registry.get(
            lr_scheduler_key.lower(), with_metadata=True
        )
        return copy.deepcopy(lr_scheduler_fn)

    @classmethod
    def get_lr_scheduler_total_steps_name(cls, lr_scheduler_key: str) -> Optional[str]:
        """Try to get the num of training steps key name for lr_scheduler if needed.

        Use the metadata `total_steps_key=...` registed in registry, or find the signature of registry `fn`,
        return the signature param key which in `lr_total_steps_key_registry` of this class.
        Return None if faield, you need registed it in lr_total_steps_key_registry.
        """

        lr_scheduler_data = cls._get_lr_scheduler_class_from_registry(lr_scheduler_key)
        lr_scheduler_metadata: Dict[str, Any] = lr_scheduler_data.get("metadata", {})
        steps_params_key = lr_scheduler_metadata.get("total_steps_key", None)
        if steps_params_key:
            return steps_params_key
        lr_scheduler_fn = lr_scheduler_data.get("fn")
        lr_scheduler_params = inspect.signature(lr_scheduler_fn).parameters

        for key in lr_scheduler_params:
            steps_params_key = cls.lr_total_steps_key_registry.get(key, None)
            if steps_params_key:
                return steps_params_key

        return steps_params_key

    def configure_optimizers(self):
        """
        Implement this method in subclasses. see :method:``lightning.pytorch.LightningModule.configure_optimizers``.
        """
        return NotImplementedError

    def configure_single_optimizer(
        self,
        optimizer: OPTIMIZER_TYPE,
        lr_scheduler: Optional[LR_SCHEDULER_TYPE] = None,
        learning_rate: Optional[float] = None,
    ) -> Union[Optimizer, Tuple[Optimizer, _TORCH_LRSCHEDULER]]:
        """Implement how optimizer and optionally learning rate schedulers should be configured."""
        optimizers_kwargs: Dict[str, Any] = {}
        if isinstance(optimizer, str):
            optimizer_fn = self._get_optimizer_class_from_registry(optimizer.lower())
        elif isinstance(optimizer, collections.abc.Callable):
            optimizer_fn = optimizer
        elif isinstance(optimizer, (tuple, list)):
            if len(optimizer) != 2:
                raise TypeError(
                    f"The tuple configuration of an optimizer input must be of length 2 with the first index"
                    f" containing a str key name from {self.available_optimizers()} and the second index containing the"
                    f" required keyword arguments to initialize the Optimizer."
                )

            if not isinstance(optimizer[0], str):
                raise TypeError(
                    f"The first value in optimizer argument tuple should be a string but got {type(optimizer[0])}."
                )

            if not isinstance(optimizer[1], dict):
                raise TypeError(
                    f"The second value in optimizer argument tuple should be of dict type but got "
                    f"{type(optimizer[1])}."
                )

            optimizer_fn: Callable = self._get_optimizer_class_from_registry(
                optimizer[0]
            )
            optimizers_kwargs: Dict[str, Any] = optimizer[1]
        else:
            raise TypeError(
                f"""Optimizer should be of type string or callable or tuple(string, dictionary)
                but got {type(optimizer)}."""
            )

        if learning_rate is not None:
            optimizers_kwargs["lr"] = learning_rate

        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer: Optimizer = optimizer_fn(model_parameters, **optimizers_kwargs)
        if lr_scheduler is not None:
            return [optimizer], [self.instantiate_lr_scheduler(optimizer, lr_scheduler)]
        return optimizer

    def instantiate_lr_scheduler(
        self, optimizer: Optimizer, _lr_scheduler: LR_SCHEDULER_TYPE
    ) -> Dict[str, Any]:
        """Initiates lr_scheduler to lighting's lr_scheduler config."""
        default_scheduler_config = DEFAULT_PL_LRCONFIG
        if isinstance(_lr_scheduler, str):
            lr_scheduler_data: Dict[
                str, Any
            ] = self._get_lr_scheduler_class_from_registry(lr_scheduler)
            lr_scheduler_fn = lr_scheduler_data.pop("fn")
            lr_scheduler_metadata: Dict[str, Any] = lr_scheduler_data.pop(
                "metadata", {}
            )
            lr_scheduler_kwargs: Dict[str, Any] = {}
            lr_scheduler_config = default_scheduler_config
            for key, value in lr_scheduler_config.items():
                lr_scheduler_config[key] = lr_scheduler_metadata.pop(key, None) or value

            # auto detect total steps for lr scheduler
            if (
                self.model
                and not getattr(self.model, "trainer", None)
                and lr_scheduler_config.get("interval") == "step"
            ):
                total_steps_params_key = self.get_lr_scheduler_total_steps_name(
                    _lr_scheduler
                )
                if (
                    total_steps_params_key
                    and lr_scheduler_kwargs.get(total_steps_params_key, None) is None
                ):
                    estimated_num_training_steps = self.get_num_training_steps()
                    if (
                        estimated_num_training_steps
                        and estimated_num_training_steps != float("inf")
                    ):
                        lr_scheduler_kwargs[
                            total_steps_params_key
                        ] = estimated_num_training_steps
        elif isinstance(lr_scheduler, collections.abc.Callable):
            lr_scheduler_data = {}
            lr_scheduler_fn = lr_scheduler
            lr_scheduler_metadata: Dict[str, Any] = None
            lr_scheduler_kwargs: Dict[str, Any] = {}
            lr_scheduler_config = default_scheduler_config

        elif isinstance(lr_scheduler, (tuple, list)):
            if len(lr_scheduler) not in [2, 3]:
                raise TypeError(
                    f"The tuple configuration of an scheduler input must be:\n"
                    f"1) Of length 2 with the first index containing a str from {self.available_lr_schedulers()} and"
                    f" the second index containing the required keyword arguments to initialize the LR Scheduler.\n"
                    f"2) Of length 3 with the first index containing a str from {self.available_lr_schedulers()} and"
                    f" the second index containing the required keyword arguments to initialize the LR Scheduler and"
                    f" the third index containing a Lightning scheduler configuration dictionary of the format"
                    f" {default_scheduler_config}. NOTE: Do not set the `scheduler` key in the"
                    f" lr_scheduler_config, it will overridden with an instance of the provided scheduler key."
                )

            if not isinstance(lr_scheduler[0], (str, collections.abc.Callable)):
                raise TypeError(
                    f"The first value in lr_scheduler argument tuple should be of type string or type Callable"
                    f" but got {type(lr_scheduler[0])}."
                )

            if not isinstance(lr_scheduler[1], dict):
                raise TypeError(
                    f"The second value in lr_scheduler argument tuple should be of type dict but got"
                    f" {type(lr_scheduler[1])}."
                )

            if len(lr_scheduler) == 3 and not isinstance(lr_scheduler[2], dict):
                raise TypeError(
                    f"The third value in lr_scheduler argument tuple should be of type dict but got"
                    f" {type(lr_scheduler[2])}."
                )

            lr_scheduler_data: Dict[
                str, Any
            ] = self._get_lr_scheduler_class_from_registry(lr_scheduler[0])
            lr_scheduler_fn = lr_scheduler_data.pop("fn")
            lr_scheduler_metadata: Dict[str, Any] = lr_scheduler_data.pop(
                "metadata", {}
            )
            lr_scheduler_kwargs: Dict[str, Any] = lr_scheduler[1]
            lr_scheduler_config = default_scheduler_config
            for key, value in lr_scheduler_config.items():
                lr_scheduler_config[key] = lr_scheduler_metadata.pop(key, None) or value
            if len(lr_scheduler) == 3:
                lr_scheduler_config.update(lr_scheduler[2])

            # auto detect total steps for lr scheduler
            if (
                self.model
                and not getattr(self.model, "trainer", None)
                and lr_scheduler_config.get("interval") == "step"
            ):
                total_steps_params_key = self.get_lr_scheduler_total_steps_name(
                    lr_scheduler[0]
                )
                if (
                    total_steps_params_key
                    and lr_scheduler_kwargs.get(total_steps_params_key, None) is None
                ):
                    estimated_num_training_steps = self.get_num_training_steps()
                    if (
                        estimated_num_training_steps
                        and estimated_num_training_steps != float("inf")
                    ):
                        lr_scheduler_kwargs[
                            total_steps_params_key
                        ] = estimated_num_training_steps

        else:
            raise TypeError(
                f"`lr_scheduler` argument should be of type string or callable or tuple(string, dictionary)"
                f" or tuple(string, dictionary, dictionary) but got {type(lr_scheduler)}."
            )

        # User can register a callable that returns a lr_scheduler_config
        # 1) If return value is an instance of _LR_Scheduler -> Add to current config and return the config.
        # 2) If return value is a dictionary, check for the lr_scheduler_config `only keys` and return the config.
        lr_scheduler: Union[_TORCH_LRSCHEDULER, Dict[str, Any]] = lr_scheduler_fn(
            optimizer, **lr_scheduler_kwargs
        )

        if isinstance(lr_scheduler, dict):
            dummy_config = default_scheduler_config
            if not all(config_key in dummy_config for config_key in lr_scheduler):
                raise ValueError(
                    f"Please make sure that your custom configuration outputs either an LR Scheduler or a scheduler"
                    f" configuration with keys belonging to {list(dummy_config.keys())}."
                )
            # If all are present, return the config
            return lr_scheduler

        # If `lr_scheduler` is not a Dict, then add it to the current config and return the config.
        lr_scheduler_config["scheduler"] = lr_scheduler
        return lr_scheduler_config

    def configure_teacher_callbacks(self):
        """Configure teacher-specific callbacks.

        Manually call this funtion to get callbacks and add them to your trainer.

        Example::

            class LitTeacher(Teacher):
                def __init__(self):
                    super().__init__()

                def setup_model(self):
                    self.model.classifier = nn.Linear(1000, 42)

                def configure_teacher_callbacks(self):
                    return [PrintCallback()]


            class PrintCallback(Callback):
                def on_train_start(self, trainer, pl_module):
                    print("Training is started!")

                def on_train_end(self, trainer, pl_module):
                    print("Training is done.")


            class LitModel(TopVirtualModule):
               def __init__(self, teacher):
                    super().__init__()
                    self.l1 = None
                    self.teacher = teacher
                    self.teacher.setup_model()


            teacher = LitTeacher()
            model = LitModel(teacher)
            t_callbacks = model.configure_teacher_callbacks()
            trainer = Trainer(accelerator="gpu", devices=2, callbacks=t_callbacks)
        """
        return []

    @property
    def estimated_num_steps_per_epoch(self) -> Union[int, float]:
        r"""
        The estimated number of steps that will ``optimizer.step()`` during training in one epoch.

        This accounts for gradient accumulation and the current trainer configuration. This might sets up your training
        dataloader if hadn't been set up already.

        .. code-block:: python

            def configure_optimizers(self):
                optimizer = ...
                stepping_batches = self.trainer.estimated_num_steps_per_epoch
                num_epoch = 10
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=stepping_batches * num_epoch)
                return [optimizer], [scheduler]

        Raises:
            MisconfigurationException:
                If estimated stepping batches cannot be computed due to different `accumulate_grad_batches`
                at different epochs.
        """

        from lightning.pytorch.utilities.rank_zero import rank_zero_info

        # infinite training
        if self.trainer.max_epochs == -1:
            return (
                float("inf") if self.trainer.max_steps == -1 else self.trainer.max_steps
            )

        if self.trainer.train_dataloader is None:
            rank_zero_info(
                "Loading `train_dataloader` to estimate number of stepping batches."
            )
            state = self.trainer.state
            self.trainer.state.fn = "fit"
            self.trainer.training = True
            self.trainer.fit_loop.setup_data()
            self.trainer.state = state

        estimated_steps = self.trainer.num_training_batches
        # iterable dataset
        if estimated_steps == float("inf"):
            return estimated_steps

        estimated_steps = math.ceil(
            estimated_steps / self.trainer.accumulate_grad_batches
        )

        return estimated_steps

    @property
    def estimated_num_epochs(self):
        return max(self.trainer.max_epochs, 1)

    def get_num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if hasattr(self.trainer, "estimated_stepping_batches"):
            return self.trainer.estimated_stepping_batches


def get_callable_name(fn_or_class: Union[Callable, object]) -> str:
    return getattr(fn_or_class, "__name__", fn_or_class.__class__.__name__).lower()


# https://github.com/Lightning-Universe/lightning-flash/blob/master/src/flash/core/utilities/apply_func.py
def normalize_callable_dict(
    fn: Union[Module, Callable, Mapping, Sequence]
) -> Union[Dict, Mapping]:
    """Normalize class/func into dict."""
    if isinstance(fn, Module):
        return ModuleDict({get_callable_name(fn): fn})
    if isinstance(fn, collections.abc.Mapping):
        return fn
    if isinstance(fn, collections.abc.Sequence):
        return {get_callable_name(f): f for f in fn}
    if callable(fn):
        return {get_callable_name(fn): fn}
    return None
