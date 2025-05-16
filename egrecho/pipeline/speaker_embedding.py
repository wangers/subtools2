# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-10)

import io
from inspect import signature
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

import requests
import torch
import torchaudio

from egrecho.core.loads import (
    ResolveModelResult,
    SerializationFn,
    resolve_pretrained_model,
)
from egrecho.data.datasets.constants import (
    AUDIO_COLUMN,
    OFFLINE_FEAT_COLUMN,
    SAMPLE_RATE_COLUMN,
    SPEAKER_COLUMN,
)
from egrecho.data.features.feature_extractor_audio import SequenceFeature
from egrecho.models.architecture.speaker import XvectorMixin, XvectorOutput
from egrecho.utils.common import alt_none
from egrecho.utils.cuda_utils import GPUManager
from egrecho.utils.imports import torchaudio_ge_2_1
from egrecho.utils.logging import get_logger

from .base import DeviceMode, PipeLine

logger = get_logger(__name__)


def load_audio(
    inputs: Union[str, bytes, io.IOBase], resample_rate: Optional[int] = None
):
    if isinstance(inputs, str):
        if inputs.startswith("http://") or inputs.startswith("https://"):
            inputs = requests.get(inputs).content
    if isinstance(inputs, bytes):
        inputs, sample_rate = io.BytesIO(inputs)

    elif isinstance(inputs, io.IOBase) or isinstance(inputs, str):
        if torchaudio_ge_2_1():
            inputs, sample_rate = torchaudio.load(inputs, backend="ffmpeg")
        else:
            inputs, sample_rate = torchaudio.backend.soundfile_backend.load(inputs)
    else:
        raise TypeError(f'Invalid audio inputs type {type(inputs)} => {inputs}')
    if resample_rate is not None and sample_rate != resample_rate:
        inputs = torchaudio.functional.resample(inputs, sample_rate, resample_rate)
    inputs = inputs[:1, ...]  # first channel.
    return inputs


class SpeakerEmbedding(PipeLine):
    candidate_audio_path_col = ("audio_path", "path", "wav_path")

    def __init__(self, model: XvectorMixin, feature_extractor, **kwargs):
        super().__init__(model=model, **kwargs)
        if hasattr(feature_extractor, "get_online_extractor") and callable(
            feature_extractor.get_online_extractor
        ):
            self.extractor_type = "offline"
        else:
            self.extractor_type = "online"
        self.feature_extractor = feature_extractor

    @classmethod
    def from_pretrained(cls, extract_kwargs: Optional[Dict[str, Any]] = None, **kwargs):
        return cls.from_sv_pretrained(extract_kwargs=extract_kwargs, **kwargs)

    def __call__(
        self,
        inputs: Union[bytes, str, torch.Tensor],
        **kwargs,
    ):
        """Extract embeddings."""
        return super().__call__(inputs, **kwargs)

    def _sanitize_parameters(self, extract_kwargs=None, **kwargs):
        forward_kwargs = {}
        if extract_kwargs is not None:
            forward_kwargs["extract_kwargs"] = extract_kwargs

        postprocess_params = {}

        return {}, forward_kwargs, postprocess_params

    def preprocess(self, inputs):
        id_meta = {}
        offline_flag = False
        if isinstance(inputs, (str, bytes, io.IOBase)):
            inputs = load_audio(inputs, self.feature_extractor.sampling_rate)
        if isinstance(inputs, dict):  # now we accept a dict sample.
            id_meta = {
                key: inputs.get(key)
                for key in ("id", SPEAKER_COLUMN)
                if inputs.get(key)
            }  # use to indicate this embedding.
            if OFFLINE_FEAT_COLUMN in inputs and self.extractor_type == "offline":
                inputs = inputs[OFFLINE_FEAT_COLUMN]
                offline_flag = True
            elif AUDIO_COLUMN in inputs:
                if not isinstance(inputs[AUDIO_COLUMN], torch.Tensor):
                    raise ValueError(
                        "When passing dict samples, `audio` key means a loaded audio tensor, but "
                        f"got type: {type(inputs[AUDIO_COLUMN])!r}, check your data."
                    )
                if not inputs.get(SAMPLE_RATE_COLUMN):
                    raise ValueError(
                        "When passing dict samples, `audio` key means a loaded audio tensor, "
                        f"It must containing a `SAMPLE_RATE_COLUMN` to indicate its sample_rate, "
                        f"but got None for that col: ({SAMPLE_RATE_COLUMN}) in keys {inputs.keys()!r}."
                    )
                if inputs[SAMPLE_RATE_COLUMN] != self.feature_extractor.sampling_rate:
                    inputs[AUDIO_COLUMN] = torchaudio.functional.resample(
                        inputs[AUDIO_COLUMN],
                        inputs[SAMPLE_RATE_COLUMN],
                        self.feature_extractor.sampling_rate,
                    )
                inputs = inputs[AUDIO_COLUMN]

            # hack audio path
            elif audio_path := next(
                (
                    inputs.get(colname)
                    for colname in self.candidate_audio_path_col
                    if inputs.get(colname) is not None
                ),
                None,
            ):
                inputs = load_audio(audio_path, self.feature_extractor.sampling_rate)

        if not isinstance(inputs, torch.Tensor):
            raise ValueError("We expect a torch tensor as input")

        ndim = inputs.ndim
        if ndim > 2 or (ndim == 2 and inputs.shape[0] > 1):
            raise ValueError(
                "We expect a single channel audio input for SpeakerEmbedingPipeline"
            )

        if (
            not offline_flag and self.extractor_type == "offline"
        ):  # offline extractor but accept audio array
            extractor = self.feature_extractor.get_online_extractor()
            processed = extractor(
                inputs,
                sampling_rate=self.feature_extractor.sampling_rate,
            )
        else:
            processed = self.feature_extractor(
                inputs,
                sampling_rate=self.feature_extractor.sampling_rate,
            )

        return {"id_meta": id_meta, **processed}

    def _forward(self, model_inputs, extract_kwargs=None):
        id_meta = model_inputs.pop("id_meta")
        extract_kwargs = extract_kwargs or {}

        # sanitize `attention_mask`.
        forward_parameters = signature(self.model.forward).parameters
        if "attention_mask" not in forward_parameters:
            if "attention_mask" in set(model_inputs):
                key = list(model_inputs.keys())[0]
                first = model_inputs[key]
                if isinstance(first, list):
                    infered_batch_size = len(first)
                else:
                    infered_batch_size = first.shape[0]
                if infered_batch_size > 1:
                    raise ValueError(
                        f"You are passing a batch_size ({infered_batch_size}) of inputs with `attention_mask` "
                        f"but `attention_mask` not in your model method :method::``extract_embedding`` signature keys "
                        f"({set(forward_parameters)}), try to set `batch_size=1` or modify your data preprocess."
                    )
                else:
                    model_inputs.pop("attention_mask", None)

        model_outputs: XvectorOutput = self.model.extract_embedding(
            **model_inputs, **extract_kwargs
        )
        return {"id_meta": id_meta, **model_outputs}

    def postprocess(self, model_outputs):
        id_meta = model_outputs.pop("id_meta")
        return {**id_meta, **model_outputs}

    @classmethod
    def load_extractor_type(cls, module_path: str):
        return super().load_module(module_path, SequenceFeature)

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
    def from_sv_pretrained(
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
