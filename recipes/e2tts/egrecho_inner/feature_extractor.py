from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torchaudio
from einops import rearrange
from einops.layers.torch import Rearrange
from lhotse.features.base import FeatureExtractor, register_extractor
from lhotse.utils import Seconds

from egrecho.data.features.feature_extractor_third_lhotse import (
    ExtLhotseFeatureExtractor,
)


@dataclass
class MelVocosConfig:
    sampling_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 100
    power: int = 1
    padding: str = "center"
    device: str = "cpu"

    def __post_init__(self):
        if self.padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "MelVocosConfig":
        return MelVocosConfig(**data)


@register_extractor
class MelVocosExtractor(
    FeatureExtractor,
):
    name = "vocos-spec"
    config_type = MelVocosConfig

    def __init__(self, config: Optional[MelVocosConfig] = None):
        super(MelVocosExtractor, self).__init__(config)
        config_dict = self.config.to_dict()
        config_dict.pop("device")
        config_dict["center"] = config_dict.pop("padding", "center") == "center"

        self.sampling_rate = config_dict.pop("sampling_rate")
        self.hop_length = self.config.hop_length
        self.num_filters = self.config.n_mels

        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sampling_rate, **config_dict
        )
        self.win_length = mel_spec.win_length
        self.extractor = torch.nn.Sequential(
            mel_spec,
            SafeLog(),
            Rearrange("... f t -> ... t f"),
        )

        # compatible for _extact_batch
        self.extractor.frame_shift = self.frame_shift

    @property
    def device(self) -> Union[str, torch.device]:
        return self.config.device

    @property
    def frame_shift(self) -> Seconds:
        return self.hop_length / self.sampling_rate

    def to(self, device: str):
        self.config.device = device

    def feature_dim(self, sampling_rate: int) -> int:
        return self.num_filters

    def extract(
        self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int
    ) -> np.ndarray:
        assert sampling_rate == self.config.sampling_rate, (
            f"Spectrogram was instantiated for sampling_rate "
            f"{self.config.sampling_rate}, but "
            f"sampling_rate={sampling_rate} was passed to extract(). "
            "Note you can use CutSet/RecordingSet.resample() to change the audio sampling rate."
        )
        is_numpy = False
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
            is_numpy = True
        if samples.ndim == 1:
            samples = samples.unsqueeze(0)
        samples = samples.to(self.device)
        if self.config.padding == "same":
            pad = self.win_length - self.hop_length
            samples = torch.nn.functional.pad(
                samples, (pad // 2, pad // 2), mode="reflect"
            )
        feats = self.extractor(samples.to(self.device))[0]
        if is_numpy:
            return feats.cpu().numpy()
        else:
            return feats

    def extract_batch(self, samples, sampling_rate, lengths=None) -> np.ndarray:
        from egrecho.data.features.lhotse_kaldi.extractor import _extract_batch

        return _extract_batch(
            self.extractor,
            samples,
            sampling_rate,
            frame_shift=self.frame_shift,
            lengths=lengths,
            device=self.device,
        )

    @staticmethod
    def mix(
        features_a: np.ndarray, features_b: np.ndarray, energy_scaling_factor_b: float
    ) -> np.ndarray:
        return np.log(
            np.maximum(
                # protection against log(0); max with EPSILON is adequate since these are energies (always >= 0)
                1e-10,
                np.exp(features_a) + energy_scaling_factor_b * np.exp(features_b),
            )
        )

    @staticmethod
    def compute_energy(features: np.ndarray) -> float:
        return float(np.sum(np.exp(features)))

    def request_vocoder(self, repo: str = "charactr/vocos-mel-24khz", **kwargs):
        device = kwargs.pop("device", "cpu")
        try:
            from vocos import Vocos
        except ImportError:
            raise ImportError("Try ``pip install vocos``")

        voc = Vocos.from_pretrained(repo)
        return voc.to(device)

    @classmethod
    def decode_audio(cls, vocos, mel):
        one_out = mel
        one_out = rearrange(one_out, "n d -> 1 d n")
        one_audio = vocos.decode(one_out)

        return one_audio


class E2TTSExtractor(ExtLhotseFeatureExtractor):
    def __init__(
        self,
        feat_conf: Optional[dict] = None,
        padding_value=0.0,
        **kwargs,
    ):
        feat_conf = feat_conf or {"feature_type": "vocos-spec"}
        super().__init__(
            feat_conf=feat_conf,
            padding_value=padding_value,
            **kwargs,
        )


class SafeLog(torch.nn.Module):
    def __init__(self, clip_val: float = 1e-7) -> None:
        super().__init__()
        self.clip_val = clip_val

    def forward(self, x, clip_val: Optional[float] = None) -> torch.Tensor:
        """
        Computes the element-wise logarithm of the input tensor with clipping to avoid near-zero values.

        Args:
            x (Tensor): Input tensor.
            clip_val (float, optional): Minimum value to clip the input tensor. Defaults to 1e-7.

        Returns:
            Tensor: Element-wise logarithm of the input tensor with clipping applied.
        """
        clip_val = clip_val or self.clip_val
        return torch.log(torch.clip(x, min=clip_val))


if __name__ == "__main__":
    print(E2TTSExtractor.available_extractors())
