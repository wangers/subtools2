# -*- coding:utf-8 -*-
# (Author: Leo 202406)

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Union

import requests
import torchaudio
from tokenizer_utils import EncodecTokenizer
from tokenizer_valle import OfflineCodesExtractor

from egrecho.data.processors.renamer import _rename_columns
from egrecho.models.valle.model import Valle
from egrecho.pipeline.base import PipeLine
from egrecho.utils.common import ObjectDict, Timer
from egrecho.utils.imports import torchaudio_ge_2_1
from egrecho.utils.types import is_tensor


def load_audio(
    inputs: Union[str, bytes, io.IOBase], resample_rate: Optional[int] = None
):
    if isinstance(inputs, str):
        if inputs.startswith("http://") or inputs.startswith("https://"):
            inputs = requests.get(inputs).content
    if isinstance(inputs, bytes):
        inputs = io.BytesIO(inputs)

    if isinstance(inputs, io.IOBase) or isinstance(inputs, str):
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


class VallePipeLine(PipeLine):
    COLS = ("text", "prompt_audio", "prompt_text")

    def __init__(self, model: Valle, **kwargs):
        feature_extractor = OfflineCodesExtractor()
        super().__init__(model=model, feature_extractor=feature_extractor, **kwargs)
        self.encodec_ = EncodecTokenizer(self.device)

        self.sampling_rate = self.encodec_.sample_rate
        self.timer = Timer()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError

    def __call__(
        self,
        inputs: Union[str, List[Dict[str, Any]]],
        prompt: tuple[str, str] = None,
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
            }
        return super().__call__(inputs, **kwargs)

    def _sanitize_parameters(
        self,
        preprocess_params=None,
        forward_params=None,
        generate_kwargs=None,
    ):
        params = {
            "forward_params": forward_params if forward_params else {},
            "generate_kwargs": generate_kwargs if generate_kwargs else {},
        }

        if preprocess_params is None:
            preprocess_params = {}
        postprocess_params = {}

        return preprocess_params, params, postprocess_params

    def preprocess(self, inputs, **kwargs):
        if isinstance(inputs, dict):
            assert all(k in inputs for k in self.COLS), inputs
            dew = inputs
        else:
            raise TypeError

        meta = {
            "text": dew["text"],
            "prompt_audio": dew["prompt_audio"],
            "prompt_text": dew["prompt_text"],
        }
        if id_ := dew.get('id'):
            meta['id'] = id_
        if tgt_path := dew.get('audio'):
            meta['ref_audio'] = tgt_path
        text = f"{dew['prompt_text']} {dew['text'] }".strip()

        encoded_text = self.tokenizer(text, **kwargs, return_tensors="pt")

        samples = load_audio(dew["prompt_audio"], self.feature_extractor.sampling_rate)
        codes = self.encodec_.encode(samples.unsqueeze(0).detach())[0][0]
        codes = codes.squeeze(0).permute(1, 0)  # (T, n_qnt)

        # codes saved in list to collate lately
        return {"codes": [codes], "meta": meta, **encoded_text}

    def _forward(self, model_inputs, generate_kwargs=None, forward_params=None):
        meta = model_inputs.pop("meta")
        codes = model_inputs.pop("codes")
        if not is_tensor(codes[0]):
            if is_tensor(codes[0][0]):
                codes = [c[0] for c in codes]
            else:
                raise TypeError(f"{type(codes)}")
        encoded_codes = self.feature_extractor(codes)
        encoded_codes["input_ids"] = encoded_codes.pop("input_features")
        model_inputs = _rename_columns(
            model_inputs,
            {"input_ids": "text_input_ids", "attention_mask": "text_attention_mask"},
        )
        # the new attention_mask needs move to device
        encoded_codes = self._ensure_tensor_on_device(encoded_codes, self.device)
        model_inputs.update(encoded_codes)
        self.model: Valle

        self.timer.reset()
        preds, preds_attention_mask = self.model.generate(
            **model_inputs, **generate_kwargs
        )
        gen_time = self.timer.elapse()

        assert preds_attention_mask.shape[1] == preds.shape[1]
        return {
            "preds": preds,
            "preds_attention_mask": preds_attention_mask,
            "meta": meta,
            'gen_time': gen_time,
        }

    def postprocess(self, model_outputs, **post_kwargs):
        meta = model_outputs.pop("meta")
        preds = model_outputs["preds"]
        preds_attention_mask = model_outputs["preds_attention_mask"]
        preds_len = preds_attention_mask.sum(-1)
        bsz = preds.shape[0]
        results = []

        if bsz == 1:
            meta = [meta]
        self.timer.reset()
        for i in range(bsz):
            codes = preds[i, : preds_len[i]]
            preds_to_dec = [
                (codes.transpose(0, 1).unsqueeze(0).to(device=self.device), None)
            ]

            sample = self.encodec_.decode(preds_to_dec)
            sample = sample[0].cpu().detach()
            gen_dur = round(sample.shape[1] / self.encodec_.sample_rate, 3)
            meta[i]['gen_dur'] = gen_dur
            outdict = ObjectDict(
                samples=sample, sampling_rate=self.encodec_.sample_rate, meta=meta[i]
            )
            results.append(outdict)
        dec_time = self.timer.elapse()
        gen_time = model_outputs.pop('gen_time')
        for r in results:
            r.meta['norm_gen_time'] = round(gen_time / bsz, 3)
        # print(f'gentime {gen_time}, dec_time {dec_time}')
        return results
