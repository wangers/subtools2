# Impleyment based on lhotse:
# https://github.com/lhotse-speech/lhotse/blob/master/lhotse/features/kaldi/extractor.py


import copy
import warnings
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, is_dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from egrecho.data.features.lhotse_kaldi.layers import (
    EPSILON,
    Seconds,
    Wav2LogFilterBank,
    Wav2LogSpec,
    Wav2MFCC,
    Wav2Spec,
)
from egrecho.utils.common import asdict_filt


def compute_num_frames_from_samples(
    num_samples: int,
    frame_shift: Seconds,
    sampling_rate: int,
) -> int:
    """
    Compute the number of frames from number of samples and frame_shift in a safe way.
    """
    window_hop = round(frame_shift * sampling_rate)
    num_frames = int((num_samples + window_hop // 2) // window_hop)
    return num_frames


FEATURE_EXTRACTORS = {}


def get_extractor_type(name: str):
    """
    Return the feature extractor type corresponding to the given name.

    :param name: specifies which feature extractor should be used.
    :return: A feature extractors type.
    """
    return FEATURE_EXTRACTORS[name]


def register_extractor(cls):
    """
    This decorator is used to register feature extractor classes in Lhotse so they can be easily created
    just by knowing their name.

    An example of usage:

    @register_extractor
    class MyFeatureExtractor: ...

    :param cls: A type (class) that is being registered.
    :return: Registered type.
    """
    FEATURE_EXTRACTORS[cls.name] = cls
    return cls


class LhotseFeat(metaclass=ABCMeta):
    """
    The base class for all feature extractors in Lhotse.
    It is initialized with a config object, specific to a particular feature extraction method.
    The config is expected to be a dataclass so that it can be easily serialized.

    All derived feature extractors must implement at least the following:

    * a ``name`` class attribute (how are these features called, e.g. 'mfcc')
    * a ``config_type`` class attribute that points to the configuration dataclass type
    * the ``extract`` method,
    * the ``frame_shift`` property.

    Feature extractors that support feature-domain mixing should additionally specify two static methods:

    * ``compute_energy``, and
    * ``mix``.

    """

    name = None
    config_type = None

    def __init__(self, config: Optional[Any] = None):
        if config is None:
            config = self.config_type()
        assert is_dataclass(
            config
        ), "The feature configuration object must be a dataclass."
        self.config = config

    @abstractmethod
    def extract(self, samples: np.ndarray, sampling_rate: int) -> np.ndarray:
        """
        Defines how to extract features using a numpy ndarray of audio samples and the sampling rate.

        :return: a numpy ndarray representing the feature matrix.
        """
        pass

    @abstractmethod
    def extract_batch(
        self,
        samples: Union[
            np.ndarray, torch.Tensor, Sequence[np.ndarray], Sequence[torch.Tensor]
        ],
        sampling_rate: int,
    ) -> Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]]:
        """
        Performs batch extraction.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def frame_shift(self) -> Seconds:
        ...

    @abstractmethod
    def feature_dim(self, sampling_rate: int) -> int:
        ...

    @property
    def device(self) -> Union[str, torch.device]:
        return "cpu"

    @staticmethod
    def mix(
        features_a: np.ndarray, features_b: np.ndarray, energy_scaling_factor_b: float
    ) -> np.ndarray:
        """
        Perform feature-domain mix of two signals, ``a`` and ``b``, and return the mixed signal.

        :param features_a: Left-hand side (reference) signal.
        :param features_b: Right-hand side (mixed-in) signal.
        :param energy_scaling_factor_b: A scaling factor for ``features_b`` energy.
            It is used to achieve a specific SNR.
            E.g. to mix with an SNR of 10dB when both ``features_a`` and ``features_b`` energies are 100,
            the ``features_b`` signal energy needs to be scaled by 0.1.
            Since different features (e.g. spectrogram, fbank, MFCC) require different combination of
            transformations (e.g. exp, log, sqrt, pow) to allow mixing of two signals, the exact place
            where to apply ``energy_scaling_factor_b`` to the signal is determined by the implementer.
        :return: A mixed feature matrix.
        """
        raise ValueError(
            'The feature extractor\'s "mix" operation is undefined. '
            "It does not support feature-domain mix, consider computing the features "
            "after, rather than before mixing the cuts."
        )

    @staticmethod
    def compute_energy(features: np.ndarray) -> float:
        """
        Compute the total energy of a feature matrix. How the energy is computed depends on a
        particular type of features.
        It is expected that when implemented, ``compute_energy`` will never return zero.

        :param features: A feature matrix.
        :return: A positive float value of the signal energy.
        """
        raise ValueError(
            'The feature extractor\'s "compute_energy" operation is undefined. '
            "It does not support feature-domain mix, consider computing the features "
            "after, rather than before mixing the cuts."
        )

    @classmethod
    def from_dict(cls, data: dict) -> "LhotseFeat":
        data = copy.deepcopy(data)
        feature_type = data.pop("feature_type")
        extractor_type = get_extractor_type(feature_type)
        # noinspection PyUnresolvedReferences
        config = extractor_type.config_type.from_dict(data)
        return extractor_type(config)

    def to_dict(self) -> Dict[str, Any]:
        d = self.config.to_dict()
        d["feature_type"] = self.name  # Insert the typename for config readability
        return d


@dataclass
class FbankConfig:
    sampling_rate: int = 16000
    frame_length: Seconds = 0.025
    frame_shift: Seconds = 0.01
    round_to_power_of_two: bool = True
    remove_dc_offset: bool = True
    preemph_coeff: float = 0.97
    window_type: str = "povey"
    dither: float = 0.0
    snip_edges: bool = False
    energy_floor: float = EPSILON
    raw_energy: bool = True
    use_energy: bool = False
    use_fft_mag: bool = False
    low_freq: float = 20.0
    high_freq: float = -400.0
    num_filters: int = 80
    num_mel_bins: Optional[int] = None  # do not use
    norm_filters: bool = False
    device: str = "cpu"

    def __post_init__(self):
        # This is to help users transition to a different Fbank implementation
        # from torchaudio.compliance.kaldi.fbank(), where the arg had a different name.
        if self.num_mel_bins is not None:
            self.num_filters = self.num_mel_bins
            self.num_mel_bins = None

        if self.snip_edges:
            warnings.warn(
                "`snip_edges` is set to True, which may cause issues in duration to num-frames conversion in Lhotse."
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict_filt(self, filt_type="none")

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "FbankConfig":
        return FbankConfig(**data)


@register_extractor
class Fbank(LhotseFeat):
    name = "kaldi-fbank"
    config_type = FbankConfig

    def __init__(self, config: Optional[FbankConfig] = None):
        super().__init__(config=config)
        config_dict = self.config.to_dict()
        config_dict.pop("device")
        self.extractor = Wav2LogFilterBank(**config_dict).to(self.device).eval()

    @property
    def device(self) -> Union[str, torch.device]:
        return self.config.device

    @property
    def frame_shift(self) -> Seconds:
        return self.config.frame_shift

    def to(self, device: str):
        self.config.device = device
        self.extractor.to(device)

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.num_filters

    def extract(
        self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int
    ) -> Union[np.ndarray, torch.Tensor]:
        assert sampling_rate == self.config.sampling_rate, (
            f"Fbank was instantiated for sampling_rate "
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

        feats = self.extractor(samples.to(self.device))[0]

        if is_numpy:
            return feats.cpu().numpy()
        else:
            return feats

    def extract_batch(
        self,
        samples: Union[
            np.ndarray, torch.Tensor, Sequence[np.ndarray], Sequence[torch.Tensor]
        ],
        sampling_rate: int,
        lengths: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]]:
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
                EPSILON,
                np.exp(features_a) + energy_scaling_factor_b * np.exp(features_b),
            )
        )

    @staticmethod
    def compute_energy(features: np.ndarray) -> float:
        return float(np.sum(np.exp(features)))


@dataclass
class MfccConfig:
    sampling_rate: int = 16000
    frame_length: Seconds = 0.025
    frame_shift: Seconds = 0.01
    round_to_power_of_two: bool = True
    remove_dc_offset: bool = True
    preemph_coeff: float = 0.97
    window_type: str = "povey"
    dither: float = 0.0
    snip_edges: bool = False
    energy_floor: float = EPSILON
    raw_energy: bool = True
    use_energy: bool = False
    use_fft_mag: bool = False
    low_freq: float = 20.0
    high_freq: float = -400.0
    num_filters: int = 23
    num_mel_bins: Optional[int] = None  # do not use
    norm_filters: bool = False
    num_ceps: int = 13
    cepstral_lifter: int = 22
    device: str = "cpu"

    def __post_init__(self):
        # This is to help users transition to a different Mfcc implementation
        # from torchaudio.compliance.kaldi.fbank(), where the arg had a different name.
        if self.num_mel_bins is not None:
            self.num_filters = self.num_mel_bins
            self.num_mel_bins = None

        if self.snip_edges:
            warnings.warn(
                "`snip_edges` is set to True, which may cause issues in duration to num-frames conversion in Lhotse."
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict_filt(self, filt_type="none")

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "MfccConfig":
        return MfccConfig(**data)


@register_extractor
class Mfcc(LhotseFeat):
    name = "kaldi-mfcc"
    config_type = MfccConfig

    def __init__(self, config: Optional[MfccConfig] = None):
        super().__init__(config=config)
        config_dict = self.config.to_dict()
        config_dict.pop("device")
        self.extractor = Wav2MFCC(**config_dict).to(self.device).eval()

    @property
    def device(self) -> Union[str, torch.device]:
        return self.config.device

    @property
    def frame_shift(self) -> Seconds:
        return self.config.frame_shift

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.num_ceps

    def extract(
        self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int
    ) -> Union[np.ndarray, torch.Tensor]:
        assert sampling_rate == self.config.sampling_rate, (
            f"Mfcc was instantiated for sampling_rate "
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

        feats = self.extractor(samples.to(self.device))[0]

        if is_numpy:
            return feats.cpu().numpy()
        else:
            return feats

    def extract_batch(
        self,
        samples: Union[
            np.ndarray, torch.Tensor, Sequence[np.ndarray], Sequence[torch.Tensor]
        ],
        sampling_rate: int,
        lengths: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]]:
        return _extract_batch(
            self.extractor,
            samples,
            sampling_rate,
            frame_shift=self.frame_shift,
            lengths=lengths,
            device=self.device,
        )


@dataclass
class SpectrogramConfig:
    sampling_rate: int = 16000
    frame_length: Seconds = 0.025
    frame_shift: Seconds = 0.01
    round_to_power_of_two: bool = True
    remove_dc_offset: bool = True
    preemph_coeff: float = 0.97
    window_type: str = "povey"
    dither: float = 0.0
    snip_edges: bool = False
    energy_floor: float = EPSILON
    raw_energy: bool = True
    use_energy: bool = False
    use_fft_mag: bool = False
    device: str = "cpu"

    def __post_init__(self):
        if self.snip_edges:
            warnings.warn(
                "`snip_edges` is set to True, which may cause issues in duration to num-frames conversion in Lhotse."
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict_filt(self, filt_type="none")

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SpectrogramConfig":
        return SpectrogramConfig(**data)


@register_extractor
class Spectrogram(LhotseFeat):
    name = "kaldi-spectrogram"
    config_type = SpectrogramConfig

    def __init__(self, config: Optional[SpectrogramConfig] = None):
        super().__init__(config=config)
        config_dict = self.config.to_dict()
        config_dict.pop("device")
        self.extractor = Wav2Spec(**config_dict).to(self.device).eval()

    @property
    def device(self) -> Union[str, torch.device]:
        return self.config.device

    @property
    def frame_shift(self) -> Seconds:
        return self.config.frame_shift

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.num_ceps

    def extract(
        self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int
    ) -> Union[np.ndarray, torch.Tensor]:
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

        feats = self.extractor(samples.to(self.device))[0]

        if is_numpy:
            return feats.cpu().numpy()
        else:
            return feats.cpu()

    def extract_batch(
        self,
        samples: Union[
            np.ndarray, torch.Tensor, Sequence[np.ndarray], Sequence[torch.Tensor]
        ],
        sampling_rate: int,
        lengths: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]]:
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
        return features_a + energy_scaling_factor_b * features_b

    @staticmethod
    def compute_energy(features: np.ndarray) -> float:
        return float(np.sum(features))


@dataclass
class LogSpectrogramConfig:
    sampling_rate: int = 16000
    frame_length: Seconds = 0.025
    frame_shift: Seconds = 0.01
    round_to_power_of_two: bool = True
    remove_dc_offset: bool = True
    preemph_coeff: float = 0.97
    window_type: str = "povey"
    dither: float = 0.0
    snip_edges: bool = False
    energy_floor: float = EPSILON
    raw_energy: bool = True
    use_energy: bool = False
    use_fft_mag: bool = False
    device: str = "cpu"

    def __post_init__(self):
        if self.snip_edges:
            warnings.warn(
                "`snip_edges` is set to True, which may cause issues in duration to num-frames conversion in Lhotse."
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict_filt(self, filt_type="none")

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "LogSpectrogramConfig":
        return LogSpectrogramConfig(**data)


@register_extractor
class LogSpectrogram(LhotseFeat):
    name = "kaldi-log-spectrogram"
    config_type = LogSpectrogramConfig

    def __init__(self, config: Optional[LogSpectrogramConfig] = None):
        super().__init__(config=config)
        config_dict = self.config.to_dict()
        config_dict.pop("device")
        self.extractor = Wav2LogSpec(**config_dict).to(self.device).eval()

    @property
    def device(self) -> Union[str, torch.device]:
        return self.config.device

    @property
    def frame_shift(self) -> Seconds:
        return self.config.frame_shift

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.num_ceps

    def extract(
        self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int
    ) -> Union[np.ndarray, torch.Tensor]:
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

        feats = self.extractor(samples.to(self.device))[0]

        if is_numpy:
            return feats.cpu().numpy()
        else:
            return feats.cpu()

    def extract_batch(
        self,
        samples: Union[
            np.ndarray, torch.Tensor, Sequence[np.ndarray], Sequence[torch.Tensor]
        ],
        sampling_rate: int,
        lengths: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]]:
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
        return features_a + energy_scaling_factor_b * features_b

    @staticmethod
    def compute_energy(features: np.ndarray) -> float:
        return float(np.sum(features))


def _extract_batch(
    extractor: LhotseFeat,
    samples: Union[
        np.ndarray, torch.Tensor, Sequence[np.ndarray], Sequence[torch.Tensor]
    ],
    sampling_rate: int,
    frame_shift: Seconds = 0.01,
    lengths: Optional[Union[np.ndarray, torch.Tensor]] = None,
    device: Union[str, torch.device] = "cpu",
) -> Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]]:
    input_is_list = False
    input_is_torch = False

    if lengths is not None:
        feat_lens = [
            compute_num_frames_from_samples(l, frame_shift, sampling_rate)
            for l in lengths
        ]
        assert isinstance(
            samples, torch.Tensor
        ), "If `lengths` is provided, `samples` must be a batched and padded torch.Tensor."
    else:
        if isinstance(samples, list):
            input_is_list = True
            pass  # nothing to do with `samples`
        elif samples.ndim > 1:
            samples = list(samples)
        else:
            # The user passed an array/tensor of shape (num_samples,)
            samples = [samples.reshape(1, -1)]

        if any(isinstance(x, torch.Tensor) for x in samples):
            input_is_torch = True

        samples = [
            torch.from_numpy(x).squeeze() if isinstance(x, np.ndarray) else x.squeeze()
            for x in samples
        ]

        feat_lens = [
            compute_num_frames_from_samples(
                num_samples=len(x),
                frame_shift=extractor.frame_shift,
                sampling_rate=sampling_rate,
            )
            for x in samples
        ]
        samples = torch.nn.utils.rnn.pad_sequence(samples, batch_first=True)

    # Perform feature extraction
    input_device = samples.device
    feats = extractor(samples.to(device))
    feats.to(input_device)
    result = [feats[i, : feat_lens[i]] for i in range(len(samples))]

    if not input_is_torch:
        result = [x.numpy() for x in result]

    # If all items are of the same shape, concatenate
    if len(result) == 1:
        if input_is_list:
            return result
        else:
            return result[0]
    elif all(item.shape == result[0].shape for item in result[1:]):
        if input_is_torch:
            return torch.stack(result, dim=0)
        else:
            return np.stack(result, axis=0)
    else:
        return result
