# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-10)

import io
from inspect import signature
from typing import Any, Dict, Optional, Union

import requests
import torch
import torchaudio

from egrecho.data.datasets.constants import (
    AUDIO_COLUMN,
    OFFLINE_FEAT_COLUMN,
    SAMPLE_RATE_COLUMN,
    SPEAKER_COLUMN,
)
from egrecho.data.features.feature_extractor_audio import SequenceFeature
from egrecho.models.architecture.speaker import XvectorMixin, XvectorOutput
from egrecho.utils.imports import torchaudio_ge_2_1

from .base import PipeLine


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
        inputs = torchaudio.functional.resample(sample_rate, resample_rate)
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
        return super().from_pretrained(extract_kwargs=extract_kwargs, **kwargs)

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
