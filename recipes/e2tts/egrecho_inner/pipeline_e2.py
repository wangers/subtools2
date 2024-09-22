# -*- coding:utf-8 -*-
# (Author: Leo 202406)

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Union

import requests
import torchaudio
from feature_extractor import E2TTSExtractor

from egrecho.data.processors.renamer import _rename_columns
from egrecho.models.e2_tts.model import E2TTS, E2TTSInferOutput
from egrecho.pipeline.base import PipeLine
from egrecho.utils.common import ObjectDict, Timer
from egrecho.utils.cuda_utils import to_device
from egrecho.utils.imports import torchaudio_ge_2_1
from egrecho.utils.types import is_tensor


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
        raise TypeError(f"Invalid audio inputs type {type(inputs)} => {inputs}")

    if resample_rate is not None and sample_rate != resample_rate:
        inputs = torchaudio.functional.resample(sample_rate, resample_rate)
    inputs = inputs[:1, ...]  # first channel.
    return inputs


class E2TTSPipeLine(PipeLine):
    COLS = ("text", "prompt_audio", "prompt_text")

    def __init__(
        self,
        model: E2TTS,
        feature_extractor: E2TTSExtractor,
        vocoder: dict = None,
        duration_predictor=None,
        **kwargs,
    ):

        super().__init__(model=model, feature_extractor=feature_extractor, **kwargs)

        if isinstance(vocoder, dict):
            self.vocoder = feature_extractor.extractor.request_vocoder(
                device=self.device, **vocoder
            )

        else:
            self.vocoder = vocoder
        self.duration_predictor = duration_predictor
        if self.duration_predictor is not None:
            self.duration_predictor = to_device(self.duration_predictor, self.device)
        self.sampling_rate = self.feature_extractor.sampling_rate
        self.timer = Timer()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError

    def __call__(
        self,
        inputs: Union[str, List[Dict[str, Any]]],
        prompt: tuple[str, str] = None,
        gen_dur_sec: Optional[float] = None,
        **kwargs,
    ):
        # single
        if isinstance(inputs, str):
            assert (
                prompt is not None
            ), f"Example infer requires a prompt of tuple (AUDIO_PATH, AUDIO_TEXT)."
            prompt_audio, prompt_text = prompt
            inputs = {
                "text": inputs,
                "prompt_audio": prompt_audio,
                "prompt_text": prompt_text,
                "gen_dur_sec": gen_dur_sec,
            }
        return super().__call__(inputs, **kwargs)

    def _sanitize_parameters(
        self,
        preprocess_params=None,
        forward_params=None,
        generate_kwargs=None,
        postprocess_params=None,
    ):
        params = {
            "forward_params": forward_params if forward_params else {},
            "generate_kwargs": generate_kwargs if generate_kwargs else {},
        }

        if preprocess_params is None:
            preprocess_params = {}
        postprocess_params = postprocess_params or {}

        return preprocess_params, params, postprocess_params

    def preprocess(self, inputs, **kwargs):
        if isinstance(inputs, dict):
            assert all(k in inputs for k in self.COLS), inputs
            if (dur_s := inputs.pop("gen_dur_sec", None)) is not None:
                inputs["gen_duration"] = int(
                    dur_s / self.feature_extractor.extractor.frame_shift
                )
            dew = inputs
        else:
            raise TypeError

        meta = {
            "text": dew["text"],
            "prompt_audio": dew["prompt_audio"],
            "prompt_text": dew["prompt_text"],
        }
        if id_ := dew.get("id"):
            meta["id"] = id_
        if tgt_path := dew.get("audio"):
            meta["ref_audio"] = tgt_path
        text = f"{dew['prompt_text']} {dew['text'] }".strip()

        encoded_text = self.tokenizer(text, **kwargs, return_tensors="pt")

        samples = load_audio(dew["prompt_audio"], self.sampling_rate)

        # codes saved in list to collate lately
        return {
            "samples": [samples],
            "meta": meta,
            "gen_duration": dew.get("gen_duration"),
            **encoded_text,
        }

    def _forward(self, model_inputs, generate_kwargs=None, forward_params=None):
        meta = model_inputs.pop("meta")
        samples = model_inputs.pop("samples")

        # unwrap batch list [[s1], [s2], ...]
        if not is_tensor(samples[0]):
            if is_tensor(samples[0][0]):
                samples = [s[0] for s in samples]
            else:
                raise TypeError(f"{type(samples)}")
        feature_inputs = self.feature_extractor(samples, self.sampling_rate)
        model_inputs = _rename_columns(
            model_inputs,
            {"input_ids": "text_input_ids", "attention_mask": "text_attention_mask"},
        )

        feature_inputs = self._ensure_tensor_on_device(feature_inputs, self.device)
        model_inputs.update(feature_inputs)
        self.model: E2TTS

        self.timer.reset()

        melouts: E2TTSInferOutput = self.model.generate(
            **model_inputs, **generate_kwargs
        )
        gen_time = self.timer.elapse()

        return {
            "melouts": melouts,
            "meta": meta,
            "gen_time": gen_time,
        }

    def postprocess(self, model_outputs, **post_kwargs):
        meta = model_outputs.pop("meta")
        melouts: E2TTSInferOutput = model_outputs["melouts"]
        with_cond = post_kwargs.pop("with_cond", False)
        mel_list = melouts.to_mel_list(with_cond)
        bsz = len(mel_list)
        results = []

        if bsz == 1:
            meta = [meta]
        self.timer.reset()
        for i in range(bsz):
            preds_to_dec = mel_list[i]
            if self.vocoder:
                sample = self.feature_extractor.extractor.decode_audio(
                    self.vocoder, preds_to_dec
                )
                sample = sample.cpu().detach()
                gen_dur = round(sample.shape[-1] / self.sampling_rate, 3)
                meta[i]["gen_dur"] = gen_dur
                outdict = ObjectDict(
                    samples=sample, sampling_rate=self.sampling_rate, meta=meta[i]
                )
            else:
                gen_dur = round(
                    preds_to_dec.shape[0]
                    * self.feature_extractor.extractor.frame_shift,
                    3,
                )
                meta[i]["gen_dur"] = gen_dur
                outdict = ObjectDict(
                    mel=preds_to_dec.cpu().detach(),
                    sampling_rate=self.sampling_rate,
                    meta=meta[i],
                )
            results.append(outdict)
        dec_time = self.timer.elapse()
        gen_time = model_outputs.pop("gen_time")
        for r in results:
            r.meta["norm_gen_time"] = round(gen_time / bsz, 3)
        # print(f'gentime {gen_time}, dec_time {dec_time}')
        return results
