# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Snowdar 2019-07-01)
#                     (updated: Leo 2023-08)

import collections
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Union

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import ScriptModule
from torch.utils.data.dataloader import DataLoader

from egrecho.core.data_builder import DataBuilder, Split
from egrecho.core.teacher import Teacher
from egrecho.data.iterable import SyncDataLoader
from egrecho.utils.cuda_utils import release_memory, to_device
from egrecho.utils.imports import is_module_available
from egrecho.utils.misc import ConfigurationException

if TYPE_CHECKING:
    from torch.utils.data import Dataset


class TopVirtualModel(pl.LightningModule):
    """
    A lightning module is related to training, val, test.

    In fit (train + validate) stage, you need to set :attr:`self.teacher`, where configures
    step logics, dataloaders, criterion, etc.
    """

    __jit_unused_properties__: List[str] = [
        "teacher"
    ] + pl.LightningModule.__jit_unused_properties__

    config_class = None
    main_input_name: str = None

    def __init__(self, *args, **kwargs):
        super().__init__()
        # a pointer to its teacher
        self._teacher: Teacher = None

    def setup(self, stage: str) -> None:
        """Hook of :method:`~lightning.pytorch.core.hooks.DataHooks.setup`.

        Called this at the beginning of fit (train + validate), validate, test, or predict. This is a good hook when
        you need to build models dynamically or adjust something about them. This hook is called on every process
        when using DDP.

        Args:
            stage: either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``

        Example:
            >>> class LitModel(TopVirtualModule):
            ...    def __init__(self):
            ...         super().__init__()
            ...         self.l1 = None
            ...
            ...     def setup(self, stage):
            ...         if stage == 'fit':
            ...             self.l1 = nn.Linear(28, 1000)
            ...             self.teacher.setup_model()
            ...
            ...
            >>> class LitTeacher(Teacher):
            ...     def __init__(self):
            ...         super().__init__()
            ...
            ...     def setup_model(self):
            ...         self.model.classifier = nn.Linear(1000, 42)
            ...
            ...     def training_step(self, batch, batch_idx):
            ...         pass

            >>> model = LitModel()
            >>> teacher = LitTeacher()
            >>> model.teacher = teacher
            >>> model.setup("fit")
            >>> assert model.l1 is not None
            >>> assert model.classifier is not None
        """
        if stage == "fit":
            self.teacher.setup()

    @property
    def teacher(self) -> Teacher:
        return self._teacher

    @teacher.setter
    def teacher(self, teacher: Teacher):
        self._teacher = teacher

    def setup_teacher(self, teacher: Teacher):
        self.teacher = teacher
        # link model first
        teacher.attach_model(self)
        self.teacher.setup_model()

    def training_step(self, *args: Any, **kwargs: Any):
        """
        Redirection to :method:`training_step` in teacher.
        """
        if self.teacher is None:
            raise ConfigurationException(
                "Module can't find a teacher to fit, please set a teacher first."
            )
        return self.teacher.training_step(*args, **kwargs)

    def validation_step(self, *args: Any, **kwargs: Any):
        """
        Redirection to :method:`training_step` in teacher.
        """
        if self.teacher is None:
            raise ConfigurationException(
                "Module can't find a teacher to fit, please set a teacher first."
            )
        return self.teacher.validation_step(*args, **kwargs)

    @classmethod
    def pipeline_out(cls, model_out: Any) -> Dict:
        """Transform output (:method:``self.forward``) to dict for pipeline. write it for your specify model."""
        if isinstance(model_out, collections.abc.Mapping):
            return dict(model_out)
        return {"output": model_out}

    def on_train_start(self) -> None:
        """Called at the very beginning of train.

        If on DDP it is called on every process

        """
        if self.teacher is None:
            raise ConfigurationException(
                "Module can't find a teacher to train, please set a teacher first."
            )
        return self.teacher.on_train_start()

    def on_train_end(self) -> None:
        """Called at the very end of train.

        If on DDP it is called on every process

        """
        if self.teacher is None:
            raise ConfigurationException(
                "Module can't find a teacher to train, please set a teacher first."
            )
        return self.teacher.on_train_end()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        Redirection to :method:`configure_optimizers` in teacher.
        """
        if self.teacher is None:
            raise ConfigurationException(
                "Module can't find a teacher to train, please set a teacher first."
            )
        return self.teacher.configure_optimizers()

    def configure_teacher_callbacks(self):
        """Configure teacher-specific callbacks.

        Manually call this funtion to get callbacks and add them to your trainer.

        Example::
            >>> class LitTeacher(Teacher):
            ...     def __init__(self):
            ...         super().__init__()
            ...
            ...     def setup_model(self):
            ...         self.model.classifier = nn.Linear(1000, 42)
            ...
            ...     def configure_teacher_callbacks(self):
            ...         return [PrintCallback()]
            ...
            ...     def training_step(self, batch, batch_idx):
            ...         pass
            ...
            ...
            >>> class PrintCallback(Callback):
            ...     def on_train_start(self, trainer, pl_module):
            ...         print("Training is started!")
            ...
            ...     def on_train_end(self, trainer, pl_module):
            ...         print("Training is done.")
            ...
            ...
            >>> class LitModel(TopVirtualModule):
            ...    def __init__(self, teacher):
            ...         super().__init__()
            ...         self.l1 = None
            ...         self.teacher = teacher
            ...         self.teacher.setup_model()
            ...
            >>> teacher = LitTeacher()
            >>> model = LitModel(teacher)
            >>> t_callbacks = model.configure_teacher_callbacks()
            >>> trainer = Trainer(accelerator="gpu", devices=2, callbacks=t_callbacks)
        """
        if self.teacher:
            return self.teacher.configure_teacher_callbacks()

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: Optional[str] = None,
        map_location: Optional[Any] = "cpu",
        hparams_file: Union[Path, str] = None,
        ignore_mismatched_sizes: bool = False,
        strict: bool = True,
        **kwargs,
    ) -> "TopVirtualModel":
        """Load pretrained model from checkpoint.

        This is raw now, to be implemented.

        Args:
            checkpoint_path: Path to checkpoint. This can also be a URL, or file-like object.
            map_location:
                MAP_LOCATION_TYPE as in torch.load(). Defaults to 'cpu'.

                If you preferring to load a checkpoint saved a GPU model
                to GPU, set it to None (not move to another GPU) or set a specified device.
            hparams_file : Path or str, optional
                Path to a .yaml file with hierarchical structure
                as in this example::

                    num_classes: 5994
                    config:
                        channels: 1024

                You most likely won't need this since Lightning will always save the
                hyperparameters to the checkpoint. However, if your checkpoint weights
                do not have the hyperparameters saved, use this method to pass in a .yaml
                file with the hparams you would like to use. These will be converted
                into a dict and passed into your Model for use.
            ignore_mismatched_sizes : bool
                Whether or not to raise an error if some of the weights from the checkpoint do not have the same size
                as the weights of the model (if for instance, you are instantiating a model with 10 labels from a
                checkpoint with 3 labels). Defaults to False.
            strict : bool, optional
                Whether to strictly enforce that the keys in checkpoint match
                the keys returned by this module's state dict. Defaults to True.
            kwargs: optional
                Any extra keyword args needed to init the model.
                Can also be used to override saved hyperparameter values.
        """
        model = cls.load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            hparams_file=hparams_file,
            strict=strict,
            **kwargs,
        )
        release_memory()
        return model

    def auto(self, layer, x):
        """It is convenient for forward-computing when layer could be None or not"""
        return layer(x) if layer is not None else x

    def get_num_params(self, only_trainable: bool = False):
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad or not only_trainable
        )

    @torch.no_grad()
    def export_onnx(
        self,
        file_path: Union[str, Path],
        input_sample: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Exports the model in ONNX format in tracing mode.

        Args:
            file_path: The path of the file the onnx model should be saved to.
            input_sample: An input for tracing. Default: None (Use self.example_input_array)
            **kwargs: Will be passed to `torch.onnx.export` function.

        NOTE:
            This general method may not appropriate for every model, you can override it for your specify model.
            If you want a Scripting onnx model, you should

        Example::

            class SimpleModel(TopVirtualModule):
                def __init__(self):
                    super().__init__()
                    self.l1 = torch.nn.Linear(in_features=64, out_features=4)

                def forward(self, x):
                    return torch.relu(self.l1(x.view(x.size(0), -1)))

            model = SimpleModel()
            input_sample = torch.randn(1, 64)
            model.export_onnx("export.onnx", input_sample, export_params=True)
        """
        if not is_module_available("onnx"):
            raise ModuleNotFoundError(
                f"`Requires `onnx` to be installed to use `{type(self).__name__}.export_onnx()`"
            )
        mode = self.training

        if input_sample is None:
            if self.example_input_array is None:
                raise ValueError(
                    "Could not export to ONNX since neither `input_sample` nor"
                    " `model.example_input_array` attribute is set."
                )
            input_sample = self.example_input_array
        input_sample = to_device(input_sample, self.device)

        torch.onnx.export(self, input_sample, file_path, **kwargs)
        self.train(mode)

    @torch.no_grad()
    def export_jit(
        self,
        file_path: Optional[Union[str, Path]] = None,
        method: Optional[str] = "script",
        input_sample: Optional[Any] = None,
        **kwargs: Any,
    ) -> Union[ScriptModule, Dict[str, ScriptModule]]:
        """
        Exports the model to a TorchScript representation for inference or saving.

        By default, compiles the entire model to a :class:`~torch.jit.ScriptModule`.
        If you prefer to use tracing, provide the argument ``method='trace'`` and ensure that either the `input_sample` argument
        is provided or the model has :attr:`example_input_array` set for tracing. To customize which modules are scripted,
        you can override this method. To return multiple modules, use a dictionary.

        Args:
            file_path (Optional[Union[str, Path]]): Path to save the TorchScript representation. Default: None (no file saved).
            method (Optional[str]): Choose between 'script' (default) and 'trace' for TorchScript compilation methods.
            input_sample (Optional[Any]): An input to be used for tracing when method is set to 'trace'.
                Default: None (uses :attr:`example_input_array`) if available.
            **kwargs (Any): Additional arguments passed to :func:`torch.jit.script` or :func:`torch.jit.trace`.

        NOTE:
            - The exported script will be set to evaluation mode.
            - It is recommended to install the latest supported version of PyTorch for using this feature without limitations.
            Refer to the :mod:`torch.jit` documentation for supported features.

        Example::

            class SimpleModel(TopVirtualModule):
                def __init__(self):
                    super().__init__()
                    self.l1 = torch.nn.Linear(in_features=64, out_features=4)

                def forward(self, x):
                    return torch.relu(self.l1(x.view(x.size(0), -1)))

            model = SimpleModel()
            model.export_jit("exported_model.pt")

        Returns:
            Union[ScriptModule, Dict[str, ScriptModule]]: The converted TorchScript representation.
        """
        mode = self.training

        if method == "script":
            with _jit_compile():
                # self.__class__.forward = torch.jit.ignore(self.__class__.forward)

                script_model = torch.jit.script(self.eval(), **kwargs)
        elif method == "trace":
            # if no example inputs are provided, try to see if model has example_input_array set
            if input_sample is None:
                if self.example_input_array is None:
                    raise ValueError(
                        "Choosing method=`trace` requires either `input_sample`"
                        " or `model.example_input_array` to be defined."
                    )
                input_sample = self.example_input_array
            input_sample = to_device(input_sample, self.device)
            example_kwarg_inputs, example_inputs = None, None
            if isinstance(input_sample, collections.abc.Mapping):
                example_kwarg_inputs = dict(input_sample)
            else:
                example_inputs = input_sample
            with _jit_compile():
                script_model = torch.jit.trace(
                    func=self.eval(),
                    example_inputs=example_inputs,
                    example_kwarg_inputs=example_kwarg_inputs,
                    **kwargs,
                )
        else:
            raise ValueError(
                f"The 'method' parameter only supports 'script' or 'trace', but value given was: {method!r}"
            )

        self.train(mode)

        if file_path is not None:
            with open(file_path, "wb") as f:
                torch.jit.save(script_model, f)

        return script_model

    def is_jitting(self):
        return torch.jit.is_scripting() or torch.jit.is_tracing()


@contextmanager
def _jit_compile() -> Generator:
    TopVirtualModel._jit_is_scripting = True
    try:
        yield
    finally:
        TopVirtualModel._jit_is_scripting = False


# def valid_dummy_input_keys(
#     model: Union[Module,TopVirtualModule], model_inputs: Iterable[str]
# ) -> Tuple[bool, List[str]]:
#     """
#     validation the input keys
#     """
#     forward_parameters = signature(model.forward).parameters

#     model_inputs_set = set(model_inputs)

#     # We are fine if config_inputs has more keys than model_inputs
#     forward_inputs_set = set(forward_parameters.keys())
#     is_ok = model_inputs_set.issubset(forward_inputs_set)

#     # Make sure the input order match (VERY IMPORTANT !!!!)
#     matching_inputs = forward_inputs_set.intersection(model_inputs_set)
#     ordered_inputs = [
#         parameter
#         for parameter in forward_parameters.keys()
#         if parameter in matching_inputs
#     ]
#     return is_ok, ordered_inputs


class DataMoudle(pl.LightningDataModule):
    """
    A simple lightning datamoudle wrapper for dataloader.

    The iterable dataset in `egrecho.data.iterable.IterabelDatasetWrapper` auto sharding samples in
    different ranks, we should load the dataset in hook: :method:`setup(self)` as this hook is
    called on every process when using DDP.

    Args:
        builder (DataBuilder):
            The data builder instance of :class:`egrecho.core.data_builder.DataBuilder`
            responsible for creating the dataset.
        batch_size (Optional[int]):
            The batch size for DataLoader. Default is None for iterable dataset.
        num_workers (int):
            The number of workers for DataLoader. Default is 0.
        prefetch_factor (Optional[int]):
            The prefetch factor for DataLoader. Default is None.
        val_num_workers (Optional[int]):
            The number of workers for validation DataLoader. Defaults to 0. If
            set None it will use the same number of workers as `num_workers`.
        pin_memory (bool):
            Whether to pin memory in DataLoader. Default is True.
        fullsync (bool):
            Whether to use `SyncDataLoader`. Default is True.
        **extra_dl_kwargs: Additional keyword arguments to pass to DataLoader.
    """

    def __init__(
        self,
        builder: DataBuilder,
        batch_size: Optional[int] = None,
        num_workers: int = 0,
        prefetch_factor: Optional[int] = None,
        val_num_workers: Optional[int] = 0,
        pin_memory: bool = True,
        fullsync: bool = True,
        **extra_dl_kwargs,
    ) -> None:
        self.data_builder = builder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_num_workers = (val_num_workers is not None) or num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.fullsync = fullsync
        self.extra_dl_kwargs = extra_dl_kwargs

        self._train_ds = None
        self._val_ds = None
        self._test_ds = None
        super().__init__()

    def setup(self, stage: str) -> None:
        self.setup_data()

    def setup_data(self) -> None:
        """
        Builds datasets and assigns dataloader func to lightning datamodule.
        """
        self.__build_dataset()
        if self.train_dataset:
            self.train_dataloader = self._train_dataloader
        if self.val_dataset:
            self.val_dataloader = self._val_dataloader
        if self.test_dataset:
            self.test_dataloader = self._test_dataloader

    def __build_dataset(self):
        datasets: dict = self.data_builder.build_dataset()

        self._train_ds = datasets.get(Split.TRAIN, None)
        self._val_ds = datasets.get(Split.VALIDATION, None)
        self._test_ds = datasets.get(Split.TEST, None)

    @property
    def train_dataset(self) -> Optional["Dataset"]:
        """This property returns the train dataset."""
        return self._train_ds

    @property
    def val_dataset(self) -> Optional["Dataset"]:
        """This property returns the validation dataset."""
        return self._val_ds

    @property
    def test_dataset(self) -> Optional["Dataset"]:
        """This property returns the test dataset."""
        return self._test_ds

    def _train_dataloader(self) -> DataLoader:
        dl_cls = SyncDataLoader if self.fullsync else DataLoader
        dataloader = dl_cls(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            **self.extra_dl_kwargs,
        )

        return dataloader

    def _val_dataloader(self) -> DataLoader:
        num_workers = self.val_num_workers
        prefetch_factor = None if num_workers < 1 else self.prefetch_factor
        dl_cls = SyncDataLoader if self.fullsync else DataLoader
        dataloader = dl_cls(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=self.pin_memory,
            **self.extra_dl_kwargs,
        )

        return dataloader

    def _test_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            **self.extra_dl_kwargs,
        )

        return dataloader
