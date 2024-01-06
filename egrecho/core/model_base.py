# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-08)

import collections
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple, Union

import torch
from torch import ScriptModule, Tensor
from torch.nn import Module

from egrecho.core.config import DataclassConfig
from egrecho.utils.cuda_utils import to_device
from egrecho.utils.imports import is_module_available
from egrecho.utils.io import auto_open


@dataclass
class ModuleConfig(DataclassConfig):
    """
    Base class for model configuration.

    """


class ModuleUtilMixin:
    def get_num_params(self, only_trainable: bool = False):
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad or not only_trainable
        )

    def auto(self, layer, x):
        """It is convenient for forward-computing when layer could be None or not"""
        return layer(x) if layer is not None else x

    @torch.jit.unused
    @property
    def device(self) -> torch.device:
        """
        The device on which of the module.
        """
        return next(self.parameters()).device

    @torch.jit.unused
    @property
    def dtype(self) -> torch.dtype:
        """
        The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        last_dtype = None
        for t in self.parameters():
            last_dtype = t.dtype
            if t.is_floating_point():
                last_dtype = t.dtype
        if last_dtype is not None:
            # if no floating dtype was found return whatever the first dtype is
            return last_dtype
        else:
            raise ValueError(f"Failed to get {self}'s dtype.")


class ModelBase(ModuleUtilMixin, Module):
    """
    A virtual backbone that provides common utilities.

    Its implementation is used to aggregate components of submodules to a model.
    """

    __jit_is_scripting = False

    config_class = None
    main_input_name: str = None

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._dummy_inputs: Optional[Union[Tensor, Tuple, Dict]] = None

    def post_init(self):
        """
        Gives a chance to perform additional operations at the end of the model's initialization process.
        """
        self.init_weights()

    def init_weights(self):
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        """
        Avoid dumplicated initialized.
        """
        if getattr(module, "_has_post_initialized", False):
            return
        self._init_weights(module)
        module._has_post_initialized = True

    def _init_weights(self, module: Module):
        """
        The specify method for initiation, override by subclasses.
        """
        ...

    @torch.jit.unused
    @property
    def dummy_inputs(self) -> Optional[Union[Tensor, Tuple, Dict]]:
        """Dummy inputs to do a forward pass in the network.

        The return type is interpreted as follows:

        -   Single tensor: It is assumed the model takes a single argument, i.e.,
            ``model.forward(model.dummy_inputs)``.
        -   Tuple: The inputs is interpreted as a sequence of positional arguments, i.e.,
            ``model.forward(*model.dummy_inputs)``.
        -   Dict: The input array represents named keyword arguments, i.e.,
            ``model.forward(**model.dummy_inputs)``.
        """
        return self._dummy_inputs

    @dummy_inputs.setter
    def dummy_inputs(self, example: Optional[Union[Tensor, Tuple, Dict]]) -> None:
        self._dummy_inputs = example

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
            input_sample: An input for tracing. Default: None (Use :attr:`dummy_inputs`)
            \**kwargs: Will be passed to :func:`torch.onnx.export`.

        NOTE:
            This general method may not appropriate for every model, you can override it for your specify model.

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
            if self.dummy_inputs is None:
                raise ValueError(
                    "Could not export to ONNX since neither `input_sample` nor"
                    " `model.dummy_inputs` attribute is set."
                )
            input_sample = self.dummy_inputs

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
        If you prefer to use tracing, provide the argument ``method='trace'`` and
        ensure that either the ``input_sample`` argument is provided or the model
        has :attr:`dummy_inputs` set for tracing. To customize which modules are scripted,
        you can override this method. To return multiple modules, use a dictionary.

        Args:
            file_path (Optional[Union[str, Path]]): Path to save the TorchScript representation.
                Default: None (no file saved).
            method (Optional[str]): Choose between 'script' (default) and 'trace' for TorchScript compilation methods.
            input_sample (Optional[Any]): An input to be used for tracing when method is set to 'trace'.
                Default: None (uses :attr:`dummy_inputs`) if available.
            \**kwargs (Any): Additional arguments passed to :func:`torch.jit.script` or :func:`torch.jit.trace`.

        NOTE:
            - The exported script will be set to evaluation mode.
            - It is recommended to install the latest supported version of PyTorch for using this feature without limitations.

            Refer to the pytorch :mod:`torch.jit` documentation for supported features.

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
                script_model = torch.jit.script(self.eval(), **kwargs)
        elif method == "trace":
            # if no example inputs are provided, try to see if model has example_input_array set
            if input_sample is None:
                if self.dummy_inputs is None:
                    raise ValueError(
                        "Choosing method=`trace` requires either `input_sample`"
                        " or `model.dummy_inputs` to be defined."
                    )
                input_sample = self.dummy_inputs
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
            with auto_open(file_path, "wb") as f:
                torch.jit.save(script_model, f)

        return script_model


@contextmanager
def _jit_compile() -> Generator:
    ModelBase.__jit_is_scripting = True
    try:
        yield
    finally:
        ModelBase.__jit_is_scripting = False
