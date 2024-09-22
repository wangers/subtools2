#!/usr/bin/env python3
# Copyright    2023                            (authors: Feiteng Li)
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

# (Author: Leo 2024-06)
# Modified from: https://github.com/lifeiteng/vall-e/blob/main/valle/data/tokenizer.py

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
from lhotse.features import FeatureExtractor
from lhotse.utils import Seconds, compute_num_frames

from egrecho.utils.text import (
    basic_cleaners,
    chinese_mandarin_cleaners,
    english_cleaners,
)

try:
    from pypinyin import Style, pinyin
    from pypinyin.style._utils import get_finals, get_initials
except Exception:
    pass

try:
    from piper_phonemize import phonemize_espeak
except Exception as ex:
    raise RuntimeError(
        f"{ex}\nPlease run\n"
        "pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html"
    )

_DEFAULT_MARKS = ';:,.!?¡¿—…"«»“”'


@dataclass(frozen=True)
class Separator:
    phone: str = "|"
    syllable: str = "-"
    word: str = " "


class G2PModel:
    """Phonemize Text."""

    def __init__(
        self,
        language="en-us",
        backend="espeak",
        punctuation_marks=_DEFAULT_MARKS,
    ) -> None:
        if backend == "espeak":
            phonemizer = PiperEspeakBackend(
                language,
            )
        elif backend in ["pypinyin", "pypinyin_initials_finals"]:
            separator = Separator()
            if language not in {"cmn", "zh"}:
                raise ValueError(
                    f"{language} is not supported for pypinyin and pypinyin_initials_finals."
                )
            phonemizer = PypinyinBackend(
                backend=backend,
                punctuation_marks=punctuation_marks + separator.word,
                separator=separator,
            )
        else:
            raise NotImplementedError(f"{backend}")

        self.backend = phonemizer

    def __call__(
        self, text, language: Optional[Union[str, List[str]]] = None, **kwargs
    ) -> List[List[str]]:
        if isinstance(text, str):
            text = [text]

        phonemized = self.backend.phonemize(text, language)
        return phonemized


def remove_encodec_weight_norm(model):
    from encodec.modules import SConv1d
    from encodec.modules.seanet import SConvTranspose1d, SEANetResnetBlock
    from torch.nn.utils import remove_weight_norm

    encoder = model.encoder.model
    for key in encoder._modules:
        if isinstance(encoder._modules[key], SEANetResnetBlock):
            remove_weight_norm(encoder._modules[key].shortcut.conv.conv)
            block_modules = encoder._modules[key].block._modules
            for skey in block_modules:
                if isinstance(block_modules[skey], SConv1d):
                    remove_weight_norm(block_modules[skey].conv.conv)
        elif isinstance(encoder._modules[key], SConv1d):
            remove_weight_norm(encoder._modules[key].conv.conv)

    decoder = model.decoder.model
    for key in decoder._modules:
        if isinstance(decoder._modules[key], SEANetResnetBlock):
            remove_weight_norm(decoder._modules[key].shortcut.conv.conv)
            block_modules = decoder._modules[key].block._modules
            for skey in block_modules:
                if isinstance(block_modules[skey], SConv1d):
                    remove_weight_norm(block_modules[skey].conv.conv)
        elif isinstance(decoder._modules[key], SConvTranspose1d):
            remove_weight_norm(decoder._modules[key].convtr.convtr)
        elif isinstance(decoder._modules[key], SConv1d):
            remove_weight_norm(decoder._modules[key].conv.conv)


class EncodecTokenizer:
    """EnCodec audio."""

    def __init__(
        self,
        device,
    ) -> None:
        # Instantiate a pretrained EnCodec model
        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(6.0)
        remove_encodec_weight_norm(model)

        self._device = device

        self.codec = model.to(device)
        self.sample_rate = model.sample_rate
        self.channels = model.channels

    @property
    def device(self):
        return self._device

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        return self.codec.encode(wav.to(self.device))

    def decode(self, frames: torch.Tensor) -> torch.Tensor:
        return self.codec.decode(frames)

    def tokenize_audio(
        self, audio: Union[Tuple[torch.Tensor, int], str]
    ) -> torch.Tensor:
        # Load and pre-process the audio waveform
        if isinstance(audio, str):
            wav, sr = torchaudio.load(audio)
        else:
            wav, sr = audio
        wav = convert_audio(wav, sr, self.sample_rate, self.channels)
        wav = wav.unsqueeze(0)

        # Extract discrete codes from EnCodec
        with torch.no_grad():
            encoded_frames = self.encode(wav)
        return encoded_frames


def parse_gpu(gpu_id: str):
    from egrecho.utils.cuda_utils import GPUManager

    gpu_id = gpu_id or "cpu"
    gpu_id = gpu_id.lower().strip()
    if gpu_id in ("auto", 'gpu'):
        return GPUManager.detect()
    elif gpu_id == "cpu":
        return gpu_id
    try:
        gpu_id = int(gpu_id)
    except Exception as e:
        pass
    return gpu_id


@dataclass
class EncodecTokeConfig:
    frame_shift: Seconds = 320.0 / 24000
    num_quantizers: int = 8
    device: str = "cpu"

    def __post_init__(self):
        self.device = parse_gpu(self.device)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "EncodecTokeConfig":
        return EncodecTokeConfig(**data)


class EncodecTokenExtractor(FeatureExtractor):
    name = "encodec"
    config_type = EncodecTokeConfig

    def __init__(self, config: Optional[Any] = None):
        super(EncodecTokenExtractor, self).__init__(config)
        self.tokenizer = EncodecTokenizer(device=self.config.device)

    def to(self, device):
        self.tokenizer.codec.to(device)
        self.config.device = device

    @property
    def device(self) -> Union[str, torch.device]:
        return self.tokenizer.device

    def extract(
        self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int
    ) -> np.ndarray:
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
        if sampling_rate != self.tokenizer.sample_rate:
            samples = convert_audio(
                samples,
                sampling_rate,
                self.tokenizer.sample_rate,
                self.tokenizer.channels,
            )
        if len(samples.shape) == 2:
            samples = samples.unsqueeze(0)
        else:
            raise ValueError()

        device = self.device
        encoded_frames = self.tokenizer.encode(samples.detach().to(device))
        codes = encoded_frames[0][0]  # [B, n_q, T]
        if True:
            duration = round(samples.shape[-1] / sampling_rate, ndigits=12)
            expected_num_frames = compute_num_frames(
                duration=duration,
                frame_shift=self.frame_shift,
                sampling_rate=sampling_rate,
            )
            assert abs(codes.shape[-1] - expected_num_frames) <= 1
            codes = codes[..., :expected_num_frames]
        return codes.cpu().squeeze(0).permute(1, 0).numpy()

    @property
    def frame_shift(self) -> Seconds:
        return self.config.frame_shift

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.num_quantizers

    def pad_tensor_list(self, tensor_list, device, padding_value=0):
        # 计算每个张量的长度
        lengths = [tensor.shape[0] for tensor in tensor_list]
        # 使用pad_sequence函数进行填充
        tensor_list = [torch.Tensor(t).to(device) for t in tensor_list]
        padded_tensor = torch.nn.utils.rnn.pad_sequence(
            tensor_list, batch_first=True, padding_value=padding_value
        )
        return padded_tensor, lengths

    def extract_batch(self, samples, sampling_rate, **kwargs) -> np.ndarray:
        samples = [wav.squeeze() for wav in samples]
        device = self.device
        samples, lengths = self.pad_tensor_list(samples, device)
        samples = samples.unsqueeze(1)

        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
        if len(samples.shape) != 3:
            raise ValueError()
        if sampling_rate != self.tokenizer.sample_rate:
            samples = [
                convert_audio(
                    wav,
                    sampling_rate,
                    self.tokenizer.sample_rate,
                    self.tokenizer.channels,
                )
                for wav in samples
            ]
            samples = torch.stack(samples, 0)  # convert samples from list to tensor
        # Extract discrete codes from EnCodec
        with torch.no_grad():
            encoded_frames = self.tokenizer.encode(samples.detach().to(device))
        encoded_frames = encoded_frames[0][0]  # [B, n_q, T]
        batch_codes = []
        for b, length in enumerate(lengths):
            codes = encoded_frames[b]
            duration = round(length / sampling_rate, ndigits=12)
            expected_num_frames = compute_num_frames(
                duration=duration,
                frame_shift=self.frame_shift,
                sampling_rate=sampling_rate,
            )
            batch_codes.append(codes[..., :expected_num_frames])
        return [codes.cpu().permute(1, 0).numpy() for codes in batch_codes]


class PiperEspeakBackend:
    def __init__(
        self,
        language="en-us",
    ) -> None:
        self.language = language

    def phonemize(
        self,
        text: List[str],
        language: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> List[str]:
        assert isinstance(text, List)
        phonemized = []
        language = language or self.language
        for _text in text:
            _text = english_cleaners(_text)
            tokens_list = phonemize_espeak(_text, language)
            tokens = []
            for t in tokens_list:
                tokens.extend(t)
            phonemized.append(tokens)
        return phonemized

    @classmethod
    def support_langs(cls):
        return ("en-us",)


class PypinyinBackend:
    """PypinyinBackend for Chinese. Most codes is referenced from espnet.
    There are two types pinyin or initials_finals, one is
    just like "ni1 hao3", the other is like "n i1 h ao3".
    """

    def __init__(
        self,
        backend="pypinyin_initials_finals",
        punctuation_marks=_DEFAULT_MARKS,
        separator=None,
    ) -> None:
        self.backend = backend
        self.punctuation_marks = punctuation_marks
        self.separator = separator or Separator()

    def phonemize(self, text: List[str], separator=None, **kwargs) -> List[str]:
        assert isinstance(text, List)
        phonemized = []
        separator = separator or self.separator
        for _text in text:
            _text = basic_cleaners(_text)
            _text = chinese_mandarin_cleaners(_text)
            _text = _text.replace(" ", separator.word)
            phones = []
            if self.backend == "pypinyin":
                for n, py in enumerate(
                    pinyin(_text, style=Style.TONE3, neutral_tone_with_five=True)
                ):
                    if all([c in self.punctuation_marks for c in py[0]]):
                        if len(phones):
                            assert phones[-1] == separator.syllable
                            phones.pop(-1)

                        phones.extend(list(py[0]))
                    else:
                        phones.extend([py[0], separator.syllable])
            elif self.backend == "pypinyin_initials_finals":
                for n, py in enumerate(
                    pinyin(_text, style=Style.TONE3, neutral_tone_with_five=True)
                ):
                    if all([c in self.punctuation_marks for c in py[0]]):
                        if len(phones):
                            assert phones[-1] == separator.syllable
                            phones.pop(-1)
                        phones.extend(list(py[0]))
                    else:
                        if py[0][-1].isalnum():
                            initial = get_initials(py[0], strict=False)
                            if py[0][-1].isdigit():
                                final = get_finals(py[0][:-1], strict=False) + py[0][-1]
                            else:
                                final = get_finals(py[0], strict=False)
                            phones.extend(
                                [
                                    initial,
                                    separator.phone,
                                    final,
                                    separator.syllable,
                                ]
                            )
                        else:
                            assert ValueError
            else:
                raise NotImplementedError

            tokens = py_to_list(
                "".join(phones).rstrip(f"{separator.word}{separator.syllable}"),
                separator,
            )
            phonemized.append(tokens)
        return phonemized


def py_to_list(phonemized: str, separator=None) -> List[str]:

    separator = separator or Separator()
    fields = []
    for word in phonemized.split(separator.word):

        pp = re.findall(r"\w+|[^\w\s]", word, re.UNICODE)
        fields.extend([p for p in pp if p != separator.phone] + [" "])
    assert len("".join(fields[:-1])) == len(phonemized) - phonemized.count(
        separator.phone
    )
    return fields[:-1]


if __name__ == "__main__":

    g2p = G2PModel(backend="pypinyin_initials_finals", language="zh")
    pybk = PypinyinBackend()
    cn = ["我爱 你 中国"]
    pyrs = pybk.phonemize(cn)
    pyrs1 = g2p(cn)
    esp = phonemize_espeak(
        "KNOT one point one five miles per hour To get up and running quickly just follow the steps below.",
        "en-us",
    )
    print(pyrs)
    print(pyrs1)
    print(esp)
    print(len(esp[0]))
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)

    samples = torch.from_numpy(np.random.random([4, 1, 1600])).type(torch.float32)
    codes_raw = model.encode(samples)

    remove_encodec_weight_norm(model)
    codes_norm = model.encode(samples)

    assert torch.allclose(codes_raw[0][0], codes_norm[0][0])
    extractor = EncodecTokenExtractor()
    samples = torch.randn(1, 24000)
    codes = extractor.extract(samples, 24000)
    print(codes.shape)  # (75, 8)
