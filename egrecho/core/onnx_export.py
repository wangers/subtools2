# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2024-01)

import collections
import inspect
import os
import re
from collections import OrderedDict
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch

from egrecho.utils.common import alt_none
from egrecho.utils.cuda_utils import release_memory, to_device
from egrecho.utils.imports import (
    _ONNX_AVAILABLE,
    _TORCH_GREATER_EQUAL_1_9,
    check_ort_requirements,
)
from egrecho.utils.logging import get_logger
from egrecho.utils.misc import ConfigurationException
from egrecho.utils.torch_utils import to_numpy

__all__ = ["OnnxExportMixin"]

logger = get_logger()
OVERRIDE_EXPORT_FORWARD = "forward_for_onnx"
DEFAULT_ATOL_FOR_VALIDATION: float = 1e-5


class OnnxConfig:

    DEFAULT_ONNX_OPSET = 11
    ATOL_FOR_VALIDATION: float = DEFAULT_ATOL_FOR_VALIDATION

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        """Return an ordered dict contains the model's input arguments name with their dynamic axis.

        Returns:
            For each output: its name associated to the axes symbolic name and the axis position within the tensor
        """
        return {}

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        """Return an ordered dict contains the model's output arguments name with their dynamic axis.

        Returns:
            For each output: its name associated to the axes symbolic name and the axis position within the tensor
        """
        return {}

    @property
    def subconfigs(self) -> Mapping[str, "OnnxConfig"]:
        """
        Return an ordered dict contains the model's subnet's onnx config.
        """
        return {}

    def parse_input_sample(self, input_sample: Union[Tuple[Any, ...], torch.Tensor]):
        """Normalizes input_sample.

        The format of ``args`` of ``torch.onnx.export`` will be transfomed to a list and a dict as format of function args,
        and applys :meth:`rename_ambiguous_inputs` to change the dict-parameter key.
        """
        input_list, input_dict = parse_input_sample(input_sample)
        input_dict = self.rename_ambiguous_inputs(input_dict)
        return input_list, input_dict

    def rename_ambiguous_inputs(self, inputs) -> Dict[str, Dict[int, str]]:
        """
        Updates the input names of the model to export.
        Override the function when the model input names are ambiguous or too generic.

        Returns:
            `Dict[str, Dict[int, str]]`: Updated inputs.
        """
        return inputs

    def ordered_inputs(self, model) -> Dict[str, Dict[int, str]]:
        """
        Re-orders the inputs using the model forward pass signature.
        """
        inputs = self.inputs
        inputs = self.rename_ambiguous_inputs(inputs)

        ordered_inputs = {}
        sig = _signature_forward(model)

        for param in sig.parameters:
            param_regex = re.compile(rf"{param}(\..*)?$")
            to_insert = []
            for name, dynamic_axes in inputs.items():
                if re.match(param_regex, name):
                    to_insert.append((name, dynamic_axes))
            # TODO: figure out a smart way of re-ordering potential nested structures.
            # to_insert = sorted(to_insert, key=lambda t: t[0])
            for name, dynamic_axes in to_insert:
                name = self.torch_to_onnx_input_map.get(name, name)
                ordered_inputs[name] = dynamic_axes
        return ordered_inputs

    def preprocess_torch_to_ort_input(
        self, input_dict: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Preprocess torch dictionary inputs to verify onnxruntime validation.

        After :meth:`parse_input_sample` and :meth:`rename_ambiguous_inputs`, torch dictionary inputs might need further fix to
        satisfy onnxruntime input names.

        Args:
            input_dict:
                Reference inputs (dictionary part) for the model.

        Returns:
            `Dict[str, Tensor]`: The mapping holding the kwargs to provide to the model's forward function
        """

        for name, val in input_dict.items():
            name = self.torch_to_onnx_input_map.get(name, name)
            input_dict[name] = val
        return input_dict

    @staticmethod
    def args_to_ort_input(
        onnx_input_names: List[str],
        input_names: List[str],
        input_dict: Dict[str, Any],
        input_list: List[Any],
    ) -> Dict[str, Any]:
        """Normalize onnxrutime inputs to dictionary."""
        odict = {}
        for k in reversed(input_names):
            val = None
            if k in input_dict:
                val = input_dict[k].cpu().numpy()
            elif len(input_list) > 0:
                val = input_list.pop().cpu().numpy()
            if k in onnx_input_names and val is not None:
                odict[k] = val
        return odict

    @property
    def torch_to_onnx_input_map(self) -> Dict[str, str]:
        """
        Dictionary mapping input names from the PyTorch model to input names from the exported ONNX model.
        Override the function when the input names and the exported ONNX input names are different.

        Returns:
            `Dict[str, str]`: A dictionary mapping the PyTorch model input names to the exported ONNX model input names.
        """
        return {}


class OnnxExportMixin:
    """
    This mixin gives models ability to be exported for deployment to ONNX format.

    Usage:
        # exporting pre-trained model to ONNX file for deployment.
        model.eval()
        model.to('cuda')  # or to('cpu') if you don't have GPU

        model.export('mymodel.onnx', [options])  # all arguments apart from `output` are optional.
    """

    def onnx_sample(self):
        raise NotImplementedError

    def get_onnx_config(self) -> OnnxConfig:
        """Sets an :class:`OnnxConfig` obj in derived implementation as default config."""
        return OnnxConfig()

    def export_onnx(
        self,
        outdir: str,
        input_sample=None,
        onnx_config: Optional[OnnxConfig] = None,
        device: Union[str, int] = "cpu",
        verbose=False,
        atol: Optional[float] = None,
        opset: Optional[int] = None,
        do_constant_folding=True,
        verify_trace: bool = False,
        export_modules_as_functions=False,
        keep_initializers_as_inputs=None,
    ):
        """
        Exports the model to the specified format. The format is inferred from the file extension of the output file.

        Args:
            outdir (str):
                Output onnx dir.
            input_sample (list or dict):
                Example input to the model's forward function. This is used to
                trace the model and export it to ONNX/TorchScript. If the model takes multiple inputs, then
                input_sample should be a list of input examples. If the model takes named inputs, then input_sample
                should be a dictionary of input examples.
            device (Union[str, int]):
                The device to use to do the export. Defaults to "cpu".
            verbose (bool):
                If True, will print out a detailed description of the model's export steps, along with
                the internal trace logs of the export process.
            atol (Optional[float], defaults to ``None``):
                If specified, the absolute difference tolerance when validating the model.
                Otherwise, the default atol for the model will be used.
            opset (Optional[int], defaults to ``None``):
                If specified, ONNX opset version to export the model with.
                Otherwise, the default opset for the given model architecture will be used.
            do_constant_folding (bool):
                If True, will execute constant folding optimization on the model's graph
                before exporting. This is ONNX specific.
            verify_trace (bool):
                If True, will verify that the model's output matches the output of the traced
                model, up to some tolerance.
            export_modules_as_functions (bool):
                If True, will export the model's submodules as functions. This is
                ONNX specific.
            keep_initializers_as_inputs (bool):
                If True, will keep the model's initializers as inputs in the onnx graph.
                This is ONNX specific.

        Returns:
            A tuple of two outputs.
            Item 0 in the output is a list of outputs, the outputs of each subnet exported.
            Item 1 in the output is a list of string descriptions. The description of each subnet exported can be
            used for logging purposes.
        """
        all_out = []
        all_descr = []
        onnx_config = alt_none(onnx_config, self.get_onnx_config())

        subnets_and_onnx_configs, opset = self.get_subnets_and_onnx_configs(
            onnx_config, opset
        )
        Path(outdir).mkdir(exist_ok=True, parents=True)
        file_path = Path(outdir) / 'model.onnx'
        for subnet_name, model_onnx_config in subnets_and_onnx_configs.items():
            model, onnx_config = model_onnx_config
            out_name = augment_filename(file_path, subnet_name)
            out, descr, out_example = export(
                model,
                onnx_config,
                out_name,
                input_sample=input_sample,
                device=device,
                verbose=verbose,
                do_constant_folding=do_constant_folding,
                opset=opset,
                verify_trace=verify_trace,
                atol=atol,
                export_modules_as_functions=export_modules_as_functions,
                keep_initializers_as_inputs=keep_initializers_as_inputs,
            )
            # Propagate input example (default scenario, may need to be overriden)
            if input_sample is not None:
                input_sample = out_example
            all_out.append(out)
            all_descr.append(descr)
            logger.info(
                f"[✓] Successfully exported {model.__class__.__name__!r} to {out_name}"
            )
        return (all_out, all_descr)

    def _prepare_for_export(self, **kwargs):
        """
        Override this method to prepare module for export. This is in-place operation.
        """
        ...

    def _export_teardown(self):
        """
        Override this method for any teardown code after export.
        """
        pass

    def get_subnets_and_onnx_configs(
        self, top_config: OnnxConfig, opset: Optional[int] = None
    ) -> Tuple[Dict[str, Tuple[torch.nn.Module, OnnxConfig]], int]:
        """
        Returns subnets and corresponding onnx configs.

        The results are formatted as a tuple of ``(Dict[KEY_NAME, (MODULE, ONNX_CONFIG)], OPSET)``:

        -   For the part containing subnets and onnx configs ``Dict[KEY_NAME, (MODULE, ONNX_CONFIG)]``,
            it finds subnets name and sub_configs via ``top_config``'s property method :meth:`subconfigs`,
            this method returns a subconfigs dict as ``Dict[KEY_NAME, ONNX_SUB_CONFIG]``,
            If the length of subconfigs is 0, it means exporting a single model,
            the returned model_config in result will be ``{"self", (self, top_config)}``.
            Otherwise returns subnets_and_onnx_configs. if the top model lacks the named module defined in subconfigs,
            that key will be ignored, and a warning will be logged.
        -   For the opset:
                -   if ``None`` is provided, OPSET will be the default version in ``top_config``,
                    i.e., the ``DEFAULT_ONNX_OPSET`` of ``top_config`` when there are no valid subnets, while the
                    max version for all valid subconfigs.
                -   If a specific version is provided, it checks and raises a
                    :class:`~egrecho.utils.misc.ConfigurationException` if that version
                    is less than the min version of subconfigs.
        """
        subconfigs = top_config.subconfigs

        default_opsets = []
        subnets_and_onnx_configs = OrderedDict()
        if len(subconfigs) == 0:
            subnets_and_onnx_configs["self"] = (self, top_config)
            default_opsets.append(top_config.DEFAULT_ONNX_OPSET)
        else:
            for subnet_name, subconfig in subconfigs.items():
                subnet = getattr(self, subnet_name, None)
                if subnet is None or not isinstance(subnet, torch.nn.Module):
                    logger.warning(
                        f"{self.__class__.__name__!r} got invalid subnet: ({subnet_name}), "
                        f"{subnet_name} might not exists or not a torch module, and export will skip {subnet_name}.\n"
                        f"### Check it if this is not intented."
                    )
                    continue
                if not isinstance(subnet, OnnxExportMixin):
                    raise ValueError(
                        f"{self.__class__.__name__!r} got unexportable subnet "
                        f"({subnet_name}: {subnet.__class__.__name__}). Hint: reconstruct subnet ({subnet.__class__.__name__}) "
                        f"by mixing with class: ({OnnxExportMixin.__class__.__qualname__})."
                    )
                subnets_and_onnx_configs[subnet_name] = (subnet, subconfig)
                default_opsets.append(subconfig.DEFAULT_ONNX_OPSET)

        # Ensure the requested opset is sufficient
        min_opset, max_opset = min(default_opsets), max(default_opsets)
        opset = alt_none(opset, max_opset)
        if opset < min_opset:
            raise ConfigurationException(
                f"Opset {opset} is not sufficient to export {self.__class__.__name__!r}. "
                f"At least {min_opset} set by {top_config.__class__.__name__} is required."
            )
        return subnets_and_onnx_configs, opset


def augment_filename(output: str, prepend: str):
    """Adds identify name to onnx file of submodules."""
    if prepend == "self":
        return output

    path, filename = os.path.split(output)
    filename = f"{prepend}-{filename}"
    return os.path.join(path, filename)


def parse_input_sample(input_sample):
    """Normalizes input_sample (``args`` of ``torch.onnx.export``) to a list and a dict as format of function args."""
    input_dict = {}
    if isinstance(input_sample, list):
        input_list = input_sample
    elif isinstance(input_sample, tuple):
        input_list = list(input_sample)
    else:
        input_list = [input_sample]
    # process possible kwargs

    if isinstance(input_list[-1], collections.abc.Mapping):
        input_dict = dict(input_list[-1])
        input_list = input_list[:-1]
    return input_list, input_dict


def _signature_forward(model) -> inspect.Signature:
    should_be_callable = getattr(model, "forward", model)
    if callable(should_be_callable):
        return inspect.signature(should_be_callable)
    raise ValueError("model has no forward method and is not callable")


def ensure_model_input_keys(model, input_keys: Iterable[str]) -> bool:
    """
    validates the map-style inputs for export.
    """
    forward_parameters = _signature_forward(model).parameters

    model_inputs_set = set(input_keys)

    # We are fine if config_inputs has more keys than model_inputs
    forward_inputs_set = set(forward_parameters.keys())
    is_ok = model_inputs_set.issubset(forward_inputs_set)
    return is_ok


def wrap_forward_method(self):
    """Gives a chance to completely override forward method to export."""
    tp = type(self)
    old_forward_method = None
    if hasattr(tp, OVERRIDE_EXPORT_FORWARD):
        forward_method = getattr(tp, OVERRIDE_EXPORT_FORWARD)
        old_forward_method = tp.forward
        tp.forward = forward_method
    else:
        forward_method = None
    return forward_method, old_forward_method


_inference_context = torch.inference_mode if _TORCH_GREATER_EQUAL_1_9 else torch.no_grad


def export(
    model: torch.nn.Module,
    config: OnnxConfig,
    outfile: str,
    input_sample=None,
    device: Union[str, int] = "cpu",
    verbose=False,
    do_constant_folding=True,
    opset=None,
    verify_trace: bool = False,
    atol: Optional[float] = None,
    export_modules_as_functions=False,
    keep_initializers_as_inputs=None,
):
    """Exports a single onnx model."""
    if not _ONNX_AVAILABLE:
        raise ModuleNotFoundError("Requires `onnx` to be installed.")
    my_args = locals().copy()
    my_args.pop("model")
    mode = model.training
    device_orig = next(model.parameters()).device

    device = alt_none(device, -1)
    if isinstance(device, int) and device < 0:
        device = "cpu"
    device = torch.device(device)

    # Set module mode
    model.to(device=device)
    model.eval()

    exportables = [m for m in model.modules() if isinstance(m, OnnxExportMixin)]

    qual_name = model.__module__ + "." + model.__class__.__qualname__
    output_descr = f"{qual_name} exported to onnx."

    opset = alt_none(opset, config.DEFAULT_ONNX_OPSET)

    try:
        # Allow user to completely override forward method to export
        forward_method, old_forward_method = wrap_forward_method(model)

        with torch.jit.optimized_execution(True), _inference_context():
            if input_sample is None:
                try:
                    input_sample = model.onnx_sample()
                except Exception:
                    input_sample = None

            if input_sample is None:
                raise MissingInputSampleException(
                    "### [x] Failed request ``onnx_sample`` when tracing onnx:\n"
                    f"### model: {model.__class__.__name__}\n"
                    f"### onnx config: {config.__class__.__name__}\n"
                    f"\t\t - [Hint] Try to implement method :meth:`onnx_sample` in this moudle or directly pass it."
                )

            input_sample = to_device(input_sample, device)

            # prepare inputs for onnx
            input_list, input_dict = config.parse_input_sample(input_sample)
            is_ok = ensure_model_input_keys(model, input_dict.keys())
            if not is_ok:
                raise InputMatchError(
                    f"### [x] Export onnx with map-inputs sample have keys: {input_dict.keys()}, "
                    f"which is invalid for model's forward keys {_signature_forward(model).parameters.keys()}."
                )
            input_sample = tuple([*input_list, input_dict])

            # Remove i/o examples from args we propagate to enclosed Exportables
            my_args.pop("outfile")
            my_args.pop("input_sample")

            # Run (posibly overridden) prepare methods before calling forward()
            for ex in exportables:
                ex._prepare_for_export(**my_args, noreplace=True)
            if isinstance(model, OnnxExportMixin):
                model._prepare_for_export(
                    output=outfile, input_sample=input_sample, **my_args
                )

            # prepare i/o names for onnx export
            inputs = config.ordered_inputs(model)
            outputs = config.outputs
            input_names = list(inputs.keys())
            output_names = list(outputs.keys())

            dynamic_axes = dict(chain(inputs.items(), config.outputs.items()))

            with _inference_context():
                output_example = model.forward(*input_list, **input_dict)
            torch.onnx.export(
                model,
                input_sample,
                outfile,
                input_names=input_names,
                output_names=output_names,
                verbose=verbose,
                do_constant_folding=do_constant_folding,
                dynamic_axes=dynamic_axes,
                opset_version=opset,
                keep_initializers_as_inputs=keep_initializers_as_inputs,
                export_modules_as_functions=export_modules_as_functions,
            )
            in_out_str = f" ort_inputs: ({', '.join(input_names)}), ort_outputs: ({', '.join(output_names)})"
            output_descr += in_out_str
        if verify_trace:

            try:
                verify_runtime(
                    config,
                    model,
                    outfile,
                    input_sample,
                    input_names,
                    onnx_named_outputs=output_names,
                    atol=atol,
                )
                logger.info(
                    f"[✓] Validation for the onnx model: {Path(outfile).as_posix()} pass"
                )
            except Exception as e:
                logger.error(
                    f"[x] Validation for the onnx model: {Path(outfile).as_posix()} error"
                )
                raise e

    finally:
        # restore model states
        if forward_method:
            type(model).forward = old_forward_method
        if isinstance(model, OnnxExportMixin):
            model._export_teardown()
        model.train(mode)
        model.to(device=device_orig)
        release_memory(model)

    return (outfile, output_descr, output_example)


def verify_runtime(
    config: OnnxConfig,
    model: torch.nn.Module,
    onnx_model: str,
    input_sample: Union[Tuple[Any, ...], torch.Tensor],
    input_names: Optional[List[str]] = None,
    onnx_named_outputs: Optional[List[str]] = None,
    atol: Optional[float] = None,
):
    """Validates the exported onnx model."""
    check_ort_requirements()
    import onnxruntime

    logger.info(f"Validating ONNX model {Path(onnx_model).as_posix()}...")
    atol = alt_none(atol, config.ATOL_FOR_VALIDATION)

    # onnx runtime session
    onnx_session_opt = onnxruntime.SessionOptions()
    onnx_session_opt.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    )
    device = next(model.parameters()).device
    provider = (
        "CUDAExecutionProvider" if device.type == 'cuda' else "CPUExecutionProvider"
    )

    sess = onnxruntime.InferenceSession(
        Path(onnx_model).as_posix(),
        sess_options=onnx_session_opt,
        providers=[provider],
    )

    # Sometimes the exported model can have more outputs than what is specified in the ONNX config because the original
    # PyTorch model has more outputs that were forgotten in the config, so we check for that.
    ort_output_names = [output.name for output in sess.get_outputs()]
    all_onnx_outputs = set(ort_output_names)
    config_outputs = set(config.outputs)

    if all_onnx_outputs != config_outputs:
        if len(all_onnx_outputs) > len(config_outputs):
            diff = all_onnx_outputs - config_outputs
        else:
            diff = config_outputs - all_onnx_outputs

        raise OutputMatchError(
            "### [x] The exported ONNX model does not have the exact same outputs as what is provided in "
            f"{config.__class__.__name__}. Difference: {', '.join(diff)}"
            "\t\t - [Hint] For the case that auto-named outputs by torch.onnx.export, Here treat it as "
            "an error, please configure a meaningful name in the onnx config."
        )

    if input_sample is None:
        try:
            input_sample = model.onnx_sample()
        except Exception:
            input_sample = None

    if input_sample is None:
        raise MissingInputSampleException(
            "### [x] Failed request ``onnx_sample`` when verifying onnx:\n"
            f"### model: {model.__class__.__name__}\n"
            f"### onnx config: {config.__class__.__name__}\n"
            f"\t\t - [Hint] Try to implement method :meth:`onnx_sample` in this pytorch model or directly pass it."
        )

    # prepare inputs for onnx
    input_sample = to_device(input_sample, device)
    input_list, input_dict = config.parse_input_sample(input_sample)
    is_ok = ensure_model_input_keys(model, input_dict.keys())
    if not is_ok:
        raise InputMatchError(
            f"### [x] Failed verify onnx:{Path(onnx_model).as_posix()} "
            f"with map-inputs sample have keys: {input_dict.keys()}, "
            f"which is invalid for model's forward keys {_signature_forward(model).parameters.keys()}."
        )
    with _inference_context():
        output_example = model.forward(*input_list, **input_dict)

    # prepare onnxruntime outputs
    input_names = alt_none(input_names, list(config.ordered_inputs(model)))
    ort_input_names = [inp.name for inp in sess.get_inputs()]
    ort_input_dict = config.preprocess_torch_to_ort_input(input_dict)
    ort_input = config.args_to_ort_input(
        ort_input_names, input_names, ort_input_dict, input_list
    )

    if isinstance(output_example, collections.abc.Mapping):
        output_example = list(output_example.values())
    elif isinstance(output_example, (tuple, list)):
        output_example = list(output_example)
    else:
        output_example = [output_example]

    # Validate the shape and values match
    shape_failures = []
    value_failures = []
    onnx_named_outputs = alt_none(onnx_named_outputs, ort_output_names)
    ort_out = sess.run(onnx_named_outputs, ort_input)
    for i, name_and_value in enumerate(zip(onnx_named_outputs, ort_out)):
        ort_out_name, ort_value = name_and_value
        expected = output_example[i]
        expected = to_numpy(expected)
        # Shape
        if not ort_value.shape == expected.shape:
            logger.error(f"[x] shape {ort_value.shape} doesn't match {expected.shape}")
            shape_failures.append((ort_out_name, i, expected.shape, ort_value.shape))
        else:
            logger.info(f"[✓] {ort_value.shape} matches {expected.shape}")

        # Values
        try:
            if not np.allclose(expected, ort_value, atol=atol):
                max_diff = np.amax(np.abs(expected - ort_value))
                logger.error(
                    f"[x] values not close enough, max diff: {max_diff} (atol: {atol})"
                )
                value_failures.append((ort_out_name, i, max_diff))
            else:
                logger.info(f"[✓] all values close (atol: {atol})")
        except Exception:
            # If shapes do not match, it is possible that the np.allclose call fails, since we raise the proper issue
            # right after, we do not do anything here.
            pass
    if shape_failures:
        msg = "\n".join(
            f"- {t[0]} torch_return_position {t[1]}: got {t[2]} (reference) and {t[3]} (ONNX)"
            for t in shape_failures
        )
        raise ShapeError(
            f"Output shapes do not match between reference model and ONNX exported model:\n{msg}"
        )

    if value_failures:
        msg = "\n".join(
            f"- {t[0]} torch_return_position {t[1]}: max diff = {t[2]}"
            for t in value_failures
        )
        atol_msg = "The maximum absolute difference between the output of the reference model and the ONNX exported "
        f"model is not within the set tolerance {atol}:\n{msg}"
        raise AtolError(atol_msg)


class MissingInputSampleException(Exception):
    pass


class OutputMatchError(Exception):
    pass


class InputMatchError(Exception):
    pass


class ShapeError(ValueError):
    pass


class AtolError(ValueError):
    pass
