# coding=utf-8
# Adapted and modified from huggingface/transformers:
# https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py
#
# Copyright 2018 The HuggingFace Inc. team.
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
#
# Copyright xmuspeech (Author: Leo 2023-10)


import collections
import os
import types
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Type, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from egrecho.core.feature_extractor import BaseFeature
from egrecho.core.loads import (
    ResolveModelResult,
    SerializationFn,
    load_module_class,
    resolve_pretrained_model,
)
from egrecho.core.module import TopVirtualModel
from egrecho.data.iterable import IterabelDatasetWrapper, Processor
from egrecho.utils.common import alt_none
from egrecho.utils.cuda_utils import (
    _BLOCKING_DEVICE_TYPES,
    GPUManager,
    _TransferableDtype,
    apply_to_collection,
)
from egrecho.utils.imports import _TORCH_GREATER_EQUAL_1_9, _TRANSFORMERS_AVAILABLE
from egrecho.utils.logging import get_logger
from egrecho.utils.types import ModelOutput, StrEnum

logger = get_logger(__name__)


class DeviceMode(StrEnum):
    FROM_MODEL = "from_model"
    AUTO = "auto"


class PipeLine(ABC):
    """Base class for other pipelines, where defines the common pipeline logics here.

    Pipeline workflow is defined as a sequence of the following operations:

        Input -> Pre-Processing -> Model Inference -> Post-Processing -> Output

    Args:
        model: inputs model
        feature_extractor: extractor
        tokenizer: tokenizer
        device: specified device

    NOTE:
        This class follows the structure of the
        `Hugging Face Pipeline <https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py>`_.
    """

    def __init__(
        self,
        model: TopVirtualModel,
        feature_extractor: Optional[BaseFeature] = None,
        tokenizer: Optional[Callable] = None,  # to do
        device: Union[str, int, torch.device, DeviceMode] = None,
        **kwargs,
    ) -> None:
        if device == DeviceMode.FROM_MODEL:
            try:
                device = model.device
            except AttributeError:
                device = next(model.parameters()).device
        else:
            device = alt_none(device, -1)
            if device == DeviceMode.AUTO:
                device = GPUManager.detect()
            elif isinstance(device, int) and device < 0:
                device = "cpu"
            model = model.to(device=device)

        self.model = model.eval()
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.device = torch.device(device)

        self.call_count = 0
        self._batch_size = kwargs.pop("batch_size", None)
        self._num_workers = kwargs.pop("num_workers", None)
        (
            self._preprocess_params,
            self._forward_params,
            self._postprocess_params,
        ) = self._sanitize_parameters(**kwargs)

    @classmethod
    def resolve_pretrained_model(
        cls,
        checkpoint: str = "last.ckpt",
        dirpath: Optional[str] = None,
        model_type: Optional[str] = None,
        feature_config: Union[str, dict] = None,
        **kwargs,
    ) -> Tuple[ResolveModelResult, Dict[str, Any]]:
        """Isolate from :meth:`from_pretrained` to check arguments errors early."""
        kwargs = kwargs.copy()
        best_k_mode: Literal["min", "max"] = kwargs.pop("best_k_mode", "min")
        version = kwargs.pop("version", "version")
        resolve_kwargs = kwargs.pop("resolve_kwargs", {})

        resolved_opt = resolve_pretrained_model(
            checkpoint,
            dirpath,
            best_k_mode=best_k_mode,
            version=version,
            **resolve_kwargs,
        )

        if isinstance(feature_config, str):
            feature_config = SerializationFn.load_file(feature_config)
        feature_config = alt_none(feature_config, resolved_opt.feature_config)
        if (
            feature_config is None
            or feature_config.get("feature_extractor_type") is None
        ):
            raise ValueError(
                f"Got invalid feature config=({feature_config}), needs a config dict with `'feature_extractor_type'` "
                "and its related configuration to instantiate extractor. Set a yaml in your ckpt's config dir"
                " (e.g., 'config/feats_config.yaml')  or "
                "passing a dict directly."
            )

        model_type = alt_none(model_type, resolved_opt.model_type)
        if model_type is None:
            raise ValueError(
                "Required model_type to instantiate model, provide it via set it (e.g, config/types.yaml) "
                "or directly passing model_type in kwargs."
            )
        resolved_opt.model_type = model_type
        resolved_opt.feature_config = feature_config

        # served as an early import checking.
        cls.load_model_type(resolved_opt.model_type)
        cls.load_extractor_type(resolved_opt.feature_config["feature_extractor_type"])

        return resolved_opt, kwargs

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str = "last.ckpt",
        dirpath: Optional[str] = None,
        model_type: Optional[str] = None,
        feature_config: Union[str, dict] = None,
        device: Union[str, int, Literal["auto", "from_model"]] = None,
        hparams_file: Optional[Union[str, Path]] = None,
        strict: bool = True,
        resolve_mode: bool = False,
        **kwargs,
    ) -> Union["PipeLine", Tuple[ResolveModelResult, Dict[str, Any]]]:
        """Initiate pretrained model, extractor and build pipeline.

        Resolve checkpoint -> instantiate model/extractor -> load checkpoint -> move to device
        -> instantiate pipeline.

        Args:
            checkpoint (str, optional):
                The file name of checkpoint to resolve, local file needs a suffix like ".ckpt" / ".pt",
                While checkpoint="best" is a preseved key means it will find `best_k_fname` which is
                a file contains `Dict[BEST_K_MODEL_PATH, BEST_K_SCORE]`, and sort by its score to
                match a best ckpt. Defaults to "last.ckpt".
            dirpath (Path or str, optional):
                The root path. Defaults to None, which means the current directory.
            model_type (str):
                model type string, if not specified, model type will be resolved from
                :meth:`resolve_pretrained_model`.
            feature_config (str or dict):
                A extractor config dict/config_file must include a key `"feature_extractor_type"` to instantiate.
                If not specified, it will be resolved from :meth:`resolve_pretrained_model`
            device (Union[str, int, Literal["auto", "from_model"]]):
                map location.
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
            strict: Whether to strictly enforce that the keys in ``checkpoint_path`` match the keys
                returned by this module's state dict.
            version (str, optional):
                The versioned subdir name. Conmmonly subdir is named as "version_0/version_1", if you specify
                the version name with a version num, it will search that version dir, otherwise choose the max number
                of version (above "version_1"). Defaults to "version".
            best_k_mode (Literal["max", "min"], optional):
                The mode for selecting the best_k checkpoint. Defaults to "min".
            resolve_kwargs (dict):
                additional kwargs passing to :func:`~egrecho.core.loads.resolve_pretrained_model`.
            load_ckpt_kwargs (dict):
                additional kwargs passing to model's :meth:`load_from_checkpoint`.
            resolve_mode (bool):
                Only returns a tuple contains resolved ckpt opt and remain kwargs,
                you need manully call :meth:`_load_ckpt`.
            \**kwargs:
                Passing remain kwargs to pipeline :meth:`__init__` and an extra keyword args placeholder.
        """

        resolved_opt, kwargs = cls.resolve_pretrained_model(
            checkpoint,
            dirpath,
            model_type=model_type,
            feature_config=feature_config,
            **kwargs,
        )
        if not resolve_mode:
            return cls._load_ckpt(
                resolved_opt,
                hparams_file=hparams_file,
                device=device,
                strict=strict,
                **kwargs,
            )
        else:
            # prepare kwargs for manully load ckpt.
            kwargs["hparams_file"] = hparams_file
            kwargs["device"] = device
            kwargs["strict"] = strict
            return resolved_opt, kwargs

    @classmethod
    def _load_ckpt(
        cls,
        resolved_opt: ResolveModelResult,
        hparams_file: Optional[Union[str, Path]] = None,
        device: Union[str, int, Literal["auto", "from_model"]] = None,
        strict: bool = True,
        **kwargs,
    ):
        """Actually loads states dict here."""
        load_ckpt_kwargs = kwargs.pop("load_ckpt_kwargs", {})
        extractor_type = cls.load_extractor_type(
            resolved_opt.feature_config["feature_extractor_type"]
        )
        model_class = cls.load_model_type(resolved_opt.model_type)
        feature_extractor = extractor_type.from_dict(resolved_opt.feature_config)
        device = alt_none(device, -1)
        if device == DeviceMode.AUTO:
            device = GPUManager.detect()
        elif device == DeviceMode.FROM_MODEL:
            raise ValueError("DO NOT SUPPORT NOW, can not del state cache")
            # device = None
        elif isinstance(device, int) and device < 0:
            device = "cpu"
        map_location = torch.device(device) if device is not None else None
        logger.info(
            f"Loading {model_class.__name__} from ckpt ({resolved_opt.checkpoint}) to device ({device or 'as_ckpt'}).",
            ranks=0,
        )

        model = model_class.from_pretrained(
            checkpoint_path=resolved_opt.checkpoint,
            map_location="cpu",
            hparams_file=hparams_file,
            strict=strict,
            **load_ckpt_kwargs,
        )

        return cls(
            model=model,
            feature_extractor=feature_extractor,
            device=map_location,
            **kwargs,
        )

    @classmethod
    def load_module(cls, module_path: str, base_module_type: Type):
        return load_module_class(module_path, base_module_type)

    @classmethod
    def load_model_type(cls, module_path: str):
        return cls.load_module(module_path, TopVirtualModel)

    @classmethod
    def load_extractor_type(cls, module_path: str):
        return cls.load_module(module_path, BaseFeature)

    @abstractmethod
    def _sanitize_parameters(self, **pipeline_parameters):
        """
        _sanitize_parameters will be called with any excessive named arguments from either `__init__` or `__call__`
        methods. It should return 3 dictionnaries of the resolved parameters used by the various `preprocess`,
        `forward` and `postprocess` methods. Do not fill dictionnaries if the caller didn't specify a kwargs. This
        let's you keep defaults in function signatures, which is more "natural".

        It is not meant to be called directly, it will be automatically called and the final parameters resolved by
        `__init__` and `__call__`
        """
        raise NotImplementedError("_sanitize_parameters not implemented")

    @abstractmethod
    def preprocess(
        self, input: Any, **preprocess_params: Dict
    ) -> Dict[str, torch.Tensor]:
        """
        [Abstract] Take care of `input` and preprocess it to a dict which will be fed to model.
        """
        raise NotImplementedError("preprocess not implemented")

    @abstractmethod
    def _forward(
        self, input_tensors: Dict[str, torch.Tensor], **forward_params: Dict
    ) -> Dict:
        """
        [Abstract] Recive dict contains tensor from :meth:`preprocess` and run model's forward.
        """
        raise NotImplementedError("_forward not implemented")

    @abstractmethod
    def postprocess(self, model_outputs: Dict, **postprocess_params: Dict) -> Any:
        """
        [Abstract] Recive dict contains tensor from :meth:`_forward` and results the final form.
        """
        raise NotImplementedError("postprocess not implemented")

    def get_inference_context(self):
        inference_context = (
            torch.inference_mode if _TORCH_GREATER_EQUAL_1_9 else torch.no_grad
        )
        return inference_context

    def forward(self, model_inputs, **forward_params):
        """Before forward: Set inference mode and move inputs to desired device.
        After forward: move model output to cpu."""
        with self.device_placement():
            inference_context = self.get_inference_context()
            with inference_context():
                model_inputs = self._ensure_tensor_on_device(
                    model_inputs, device=self.device
                )

                model_outputs = self._forward(model_inputs, **forward_params)
                model_outputs = self._ensure_tensor_on_device(
                    model_outputs, device=torch.device("cpu")
                )

        return model_outputs

    @contextmanager
    def device_placement(self):
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        yield

    def _ensure_tensor_on_device(self, inputs, device: torch.device):
        def move_to(data: Any) -> Any:
            kwargs = {}
            # Don't issue non-blocking transfers to CPU
            # Same with MPS due to a race condition bug: https://github.com/pytorch/pytorch/issues/83015
            if isinstance(data, torch.Tensor):
                if device.type not in _BLOCKING_DEVICE_TYPES:
                    kwargs["non_blocking"] = True
                if device == torch.device("cpu") and data.dtype in {
                    torch.float16,
                    torch.bfloat16,
                }:
                    data = data.float()
            data_output = data.to(device, **kwargs)
            if data_output is not None:
                return data_output
            return data

        inputs = apply_to_collection(inputs, dtype=_TransferableDtype, function=move_to)
        return inputs

    def get_iterator(
        self,
        inputs,
        num_workers: int,
        batch_size: int,
        preprocess_params,
        forward_params,
        postprocess_params,
    ):
        if _not_map_inputs(inputs):
            if num_workers > 1:
                if not isinstance(inputs, (IterabelDatasetWrapper, Processor)):
                    logger.warning(
                        "For iterable dataset using num_workers>1 is likely to result"
                        " in errors since everything is iterable, setting `num_workers=1`"
                        " to guarantee correctness, or use `egrecho.data.IterabelDatasetWrapper`"
                        " to ensure worker partition.",
                        ranks=0,
                    )
                    num_workers = 1
            dataset = PipelineIterator(inputs, self.preprocess, preprocess_params)

        else:
            dataset = PipelineDataset(inputs, self.preprocess, preprocess_params)
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            # logger.info(
            #     "Disabling tokenizer parallelism, we're using DataLoader multithreading already"
            # )
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        feature_extractor = self.feature_extractor
        collate_fn = (
            no_collate_fn
            if batch_size == 1
            else pad_collate_fn(self.tokenizer, feature_extractor)
        )

        dataloader = DataLoader(
            dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )
        model_iterator = PipelineIterator(
            dataloader, self.forward, forward_params, loader_batch_size=batch_size
        )
        final_iterator = PipelineIterator(
            model_iterator, self.postprocess, postprocess_params
        )
        return final_iterator

    def __call__(self, inputs, *args, num_workers=None, batch_size=None, **kwargs):
        if args:
            logger.warning(f"Ignoring args : {args}")

        if num_workers is None:
            if self._num_workers is None:
                num_workers = 0
            else:
                num_workers = self._num_workers
        if batch_size is None:
            if self._batch_size is None:
                batch_size = 1
            else:
                batch_size = self._batch_size

        (
            preprocess_params,
            forward_params,
            postprocess_params,
        ) = self._sanitize_parameters(**kwargs)

        # Fuse __init__ params and __call__ params without modifying the __init__ ones.
        preprocess_params = {**self._preprocess_params, **preprocess_params}
        forward_params = {**self._forward_params, **forward_params}
        postprocess_params = {**self._postprocess_params, **postprocess_params}

        self.call_count += 1
        if self.call_count > 10 and self.device.type == "cuda":
            warnings.warn(
                "You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a"
                " dataset",
                UserWarning,
            )

        is_dataset = isinstance(inputs, Dataset)
        is_generator = isinstance(inputs, types.GeneratorType)
        is_list = isinstance(inputs, list)

        is_iterable = is_dataset or is_generator or is_list

        if is_list:
            final_iterator = self.get_iterator(
                inputs,
                num_workers,
                batch_size,
                preprocess_params,
                forward_params,
                postprocess_params,
            )
            outputs = list(final_iterator)
            return outputs

        elif is_iterable:
            return self.get_iterator(
                inputs,
                num_workers,
                batch_size,
                preprocess_params,
                forward_params,
                postprocess_params,
            )
        else:
            return self.run_single(
                inputs, preprocess_params, forward_params, postprocess_params
            )

    def run_single(self, inputs, preprocess_params, forward_params, postprocess_params):
        model_inputs = self.preprocess(inputs, **preprocess_params)

        model_outputs = self.forward(model_inputs, **forward_params)
        outputs = self.postprocess(model_outputs, **postprocess_params)
        return outputs

    def to(self, device: Union[str, int, torch.device]):
        """Send pipeline to `device`"""
        device = torch.device(device)
        self.model.to(device)

        self.device = device

        return self


class PipelineDataset(Dataset):
    def __init__(self, dataset, process, params):
        self.dataset = dataset
        self.process = process
        self.params = params

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = self.dataset[i]
        processed = self.process(item, **self.params)
        return processed


class PipelineIterator(IterableDataset):
    def __init__(self, loader, infer, params, loader_batch_size=None):
        """
        Initializes the PipelineIterator.

        This iterator is roughly equivalent to the following loop::

            for item in loader:
                yield infer(item, **params)

        Args:
            loader (torch.utils.data.DataLoader or any iterator):
                The iterator that will be used to apply `infer` on.
            infer (callable):
                The function to apply to each element of `loader`.
            params (dict):
                The parameters passed to `infer` along with every item.
            loader_batch_size (int, optional):
                If specified, the items of `loader` are supposed to come as batches,
                and are loader_batched here making it roughly behave as::

                    for items in loader:
                        for i in loader_batch_size:
                            item = items[i]
                            yield infer(item, **params)
        """
        self.loader = loader
        self.infer = infer
        self.params = params
        if loader_batch_size == 1:
            # Let's spare some time by deactivating altogether
            loader_batch_size = None
        self.loader_batch_size = loader_batch_size

        # Internal bookkeeping
        self._loader_batch_index = None
        self._loader_batch_data = None

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self.iterator = iter(self.loader)
        return self

    def loader_batch_item(self):
        """
        Return item located at `loader_batch_index` within the current `loader_batch_data`.
        """
        if isinstance(self._loader_batch_data, torch.Tensor):
            # Batch data is simple tensor, just fetch the slice
            result = self._loader_batch_data[self._loader_batch_index]
        else:
            # Batch data is assumed to be BaseModelOutput (or dict)
            loader_batched = {}
            for k, element in self._loader_batch_data.items():
                if _is_model_out(element):
                    # Convert ModelOutput to tuple first
                    element = element.to_tuple()
                    if isinstance(element[0], torch.Tensor):
                        loader_batched[k] = tuple(
                            el[self._loader_batch_index].unsqueeze(0) for el in element
                        )
                    elif isinstance(element[0], np.ndarray):
                        loader_batched[k] = tuple(
                            np.expand_dims(el[self._loader_batch_index], 0)
                            for el in element
                        )
                    continue
                if k in {
                    "hidden_states",
                    "past_key_values",
                    "attentions",
                } and isinstance(element, tuple):
                    # Those are stored as lists of tensors so need specific unbatching.
                    if isinstance(element[0], torch.Tensor):
                        loader_batched[k] = tuple(
                            el[self._loader_batch_index].unsqueeze(0) for el in element
                        )
                    elif isinstance(element[0], np.ndarray):
                        loader_batched[k] = tuple(
                            np.expand_dims(el[self._loader_batch_index], 0)
                            for el in element
                        )
                    continue
                if element is None:
                    # This can happen for optional data that get passed around
                    loader_batched[k] = None
                elif isinstance(element[self._loader_batch_index], torch.Tensor):
                    # Take correct batch data, but make it looked like batch_size=1
                    # For compatibility with other methods within transformers

                    loader_batched[k] = element[self._loader_batch_index].unsqueeze(0)
                elif isinstance(element[self._loader_batch_index], np.ndarray):
                    # Take correct batch data, but make it looked like batch_size=1
                    # For compatibility with other methods within transformers
                    loader_batched[k] = np.expand_dims(
                        element[self._loader_batch_index], 0
                    )
                else:
                    # This is typically a list, so no need to `unsqueeze`.
                    loader_batched[k] = element[self._loader_batch_index]
            # Recreate the element by reusing the original class to make it look
            # batch_size=1
            result = self._loader_batch_data.__class__(loader_batched)
        self._loader_batch_index += 1
        return result

    def __next__(self):
        if (
            self._loader_batch_index is not None
            and self._loader_batch_index < self.loader_batch_size
        ):
            # We are currently unrolling a batch so we just need to return
            # the current item within a batch
            return self.loader_batch_item()

        # We're out of items within a batch
        item = next(self.iterator)
        processed = self.infer(item, **self.params)
        # We now have a batch of "inferred things".
        if self.loader_batch_size is not None:
            # Try to infer the size of the batch
            if isinstance(processed, torch.Tensor):
                first_tensor = processed
            else:
                key = list(processed.keys())[0]
                first_tensor = processed[key]
            if isinstance(first_tensor, list):
                observed_batch_size = len(first_tensor)
            else:
                observed_batch_size = first_tensor.shape[0]
            if 0 < observed_batch_size < self.loader_batch_size:
                # could be last batch so we can't unroll as many
                # elements.
                self.loader_batch_size = observed_batch_size
            # Setting internal index to unwrap the batch
            self._loader_batch_data = processed
            self._loader_batch_index = 0
            return self.loader_batch_item()
        else:
            # We're not unrolling batches
            return processed


def no_collate_fn(items):
    if len(items) != 1:
        raise ValueError("This collate_fn is meant to be used with batch_size=1")
    return items[0]


def _pad(items, key, padding_value, padding_side):
    """Apply padding on the items of shape with [B, T, ...]."""
    batch_size = len(items)
    if isinstance(items[0][key], torch.Tensor):
        # Others include `attention_mask` etc...
        shape = items[0][key].shape
        dim = len(shape)
        if key in ["pixel_values", "image"]:
            # This is probable image so padding shouldn't be necessary
            # B, C, H, W
            return torch.cat([item[key] for item in items], dim=0)
        elif dim == 4 and key == "input_features":
            # this is probably a mel spectrogram batched
            return torch.cat([item[key] for item in items], dim=0)
        max_length = max(item[key].shape[1] for item in items)
        min_length = min(item[key].shape[1] for item in items)
        dtype = items[0][key].dtype

        if dim == 2:
            if max_length == min_length:
                # Bypass for `ImageGPT` which doesn't provide a padding value, yet
                # we can consistently pad since the size should be matching
                return torch.cat([item[key] for item in items], dim=0)
            tensor = torch.zeros((batch_size, max_length), dtype=dtype) + padding_value
        elif dim == 3:
            tensor = (
                torch.zeros((batch_size, max_length, shape[-1]), dtype=dtype)
                + padding_value
            )

        for i, item in enumerate(items):
            if dim == 2:
                if padding_side == "left":
                    tensor[i, -len(item[key][0]) :] = item[key][0].clone()
                else:
                    tensor[i, : len(item[key][0])] = item[key][0].clone()
            elif dim == 3:
                if padding_side == "left":
                    tensor[i, -len(item[key][0]) :, :] = item[key][0].clone()
                else:
                    tensor[i, : len(item[key][0]), :] = item[key][0].clone()
        return tensor
    else:
        return [item[key] for item in items]


def pad_collate_fn(tokenizer, feature_extractor):
    # Tokenizer
    t_padding_side = None
    # Feature extractor
    f_padding_side = None
    if tokenizer is None and feature_extractor is None:
        raise ValueError(
            "Pipeline without tokenizer or feature_extractor cannot do batching"
        )
    if tokenizer is not None:
        if tokenizer.pad_token_id is None:
            raise ValueError(
                "Pipeline with tokenizer without pad_token cannot do batching. You can try to set it with "
                "`pipe.tokenizer.pad_token_id = model.config.eos_token_id`."
            )
        else:
            t_padding_value = tokenizer.pad_token_id
            t_padding_side = tokenizer.padding_side
    if feature_extractor is not None:
        # Feature extractor can be images, where no padding is expected
        f_padding_value = getattr(feature_extractor, "padding_value", None)
        f_padding_side = getattr(feature_extractor, "padding_side", None)

    if (
        t_padding_side is not None
        and f_padding_side is not None
        and t_padding_side != f_padding_side
    ):
        raise ValueError(
            f"The feature extractor, and tokenizer don't agree on padding side {t_padding_side} != {f_padding_side}"
        )
    padding_side = "right"
    if t_padding_side is not None:
        padding_side = t_padding_side
    if f_padding_side is not None:
        padding_side = f_padding_side

    def inner(items):
        keys = set(items[0].keys())
        for item in items:
            if set(item.keys()) != keys:
                raise ValueError(
                    f"The elements of the batch contain different keys. Cannot batch them ({set(item.keys())} !="
                    f" {keys})"
                )
        # input_values, input_pixels, input_ids, ...
        padded = {}
        for key in keys:
            if key in {"input_ids"}:
                # ImageGPT uses a feature extractor
                if tokenizer is None and feature_extractor is not None:
                    _padding_value = f_padding_value
                else:
                    _padding_value = t_padding_value
            elif key in {"input_values", "pixel_values", "input_features"}:
                _padding_value = f_padding_value
            elif key in {"p_mask", "special_tokens_mask"}:
                _padding_value = 1
            elif key in {"attention_mask", "token_type_ids"}:
                _padding_value = 0
            else:
                # This is likely another random key maybe even user provided
                _padding_value = 0
            padded[key] = _pad(items, key, _padding_value, padding_side)
        return padded

    return inner


def _not_map_inputs(inputs):
    return (
        not isinstance(inputs, collections.abc.Sized)
        or isinstance(inputs, IterableDataset)
        or (hasattr(inputs, "is_lazy") and not inputs.is_lazy)
    )


def _is_model_out(elem):
    if isinstance(elem, ModelOutput):
        return True
    if _TRANSFORMERS_AVAILABLE:
        import transformers

        return isinstance(elem, transformers.utils.ModelOutput)
    return False
