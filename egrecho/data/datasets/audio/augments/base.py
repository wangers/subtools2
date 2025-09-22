# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Leo 2023-01-11)
# Base class for audio signal augment.

import copy
import functools
import re
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import torch
import torchaudio

from egrecho.data.datasets.audio.augments.dsp import AudioClip
from egrecho.data.datasets.audio.augments.transforms import get_or_create_resampler
from egrecho.utils.common import ObjectDict, asdict_filt
from egrecho.utils.misc import rich_exception_info

_DEFAULT_INIT_P_RE = re.compile(r"\,*\s*init_p=\d*[1-9]\.?\d*|\,*\s*init_p=None")


def _patch_repr(repr_func):
    @functools.wraps(repr_func)
    def wrapper(*args, **kwargs):
        repr_string = repr_func(*args, **kwargs)
        return _DEFAULT_INIT_P_RE.sub("", repr_string)

    return wrapper


class SignalPerturb(torch.nn.Module):
    """
    Base class for signal perturbs.
    """

    KNOWN_PERTURBS = ObjectDict()

    def __init__(self):
        super().__init__()
        self._p = 1.0

    def __init_subclass__(cls, **kwargs):
        name = cls.name if hasattr(cls, "name") else cls.__name__
        assert (
            name not in SignalPerturb.KNOWN_PERTURBS
        ), f"Failed register, perturb name:{name} is already known."
        SignalPerturb.KNOWN_PERTURBS[name] = cls

        super().__init_subclass__(**kwargs)

    def forward(
        self,
        samples: torch.Tensor = None,
        lengths: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        sample_rate: Optional[int] = None,
        **kwargs,
    ) -> ObjectDict:
        """
        This base interface is designed to support additional args in `key=value` style.

        Arguments
        ---------
        samples:
            Waveforms tensor with shape of `[batch, channel, time]` is compatible for all the derived classes.
        lengths:
            Valid lengths `[batch]`

        Returns
        -------
            A dict contains perturbed samples.
            i.e. `output_samples = output.samples` or `output_samples = output['samples']`.
        """
        # Turn down in eval stage and handles the perturb probability.
        if not self.training or torch.rand(1) > self.p:
            return ObjectDict(
                samples=samples,
                lengths=lengths,
                targets=targets,
                target_lengths=target_lengths,
                sample_rate=sample_rate,
                **kwargs,
            )

        with torch.no_grad():
            output = self.apply(
                samples=samples.clone(),
                lengths=lengths,
                targets=targets,
                target_lengths=target_lengths,
                sample_rate=sample_rate,
                **kwargs,
            )
        return output

    def apply(
        self,
        samples: torch.Tensor = None,
        lengths: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        sample_rate: Optional[int] = None,
        **kwargs,
    ) -> ObjectDict:
        """
        Apply transform.

        To be implemented in derived classes.
        """
        raise NotImplementedError

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, p):
        self._p = p

    @staticmethod
    def from_dict(data: dict) -> "SignalPerturb":
        data = copy.deepcopy(data)
        cls = SignalPerturb.get_register_class(data.pop("name"))
        return cls(**data)

    @staticmethod
    def get_register_class(name):
        assert name in SignalPerturb.KNOWN_PERTURBS, f"Unknown transform type: {name}."
        return SignalPerturb.KNOWN_PERTURBS[name]


@dataclass(unsafe_hash=True)
class SinglePerturb(SignalPerturb):
    name: ClassVar[str]

    def __post_init__(self):
        super().__init__()
        # Want to hide defalut param:init_p  in func(__repr__) of dataclass
        patch_repr = _patch_repr(type(self).__repr__)
        setattr(type(self), "__repr__", patch_repr)

    def to_dict(self, filt_type="default") -> dict:
        d = asdict_filt(self, filt_type=filt_type)
        cls = type(self)
        if self.p < 1:
            d.update({"init_p": self.p})
        name = cls.name if hasattr(cls, "name") else cls.__name__
        return {**d, "name": name}

    @abstractmethod
    def apply(self, **kwargs) -> ObjectDict:
        ...


@dataclass(unsafe_hash=True)
class MultiPerturb(SinglePerturb):
    name: ClassVar[str]
    perturbs: List[Union[Dict, SignalPerturb]]

    def __post_init__(self):
        super().__post_init__()
        perturbs = []
        for perturb in self.perturbs:
            if isinstance(perturb, dict):
                perturbs.append(SignalPerturb.from_dict(perturb))
            elif isinstance(perturb, SignalPerturb):
                perturbs.append(copy.deepcopy(perturb))
            else:
                raise ValueError(f"init `MultiPerturb` failed, position:{perturb}.")
        # self.perturbs = list(map(lambda perturb: \
        #     SignalPerturb.from_dict(perturb) if isinstance(perturb, dict))\
        #     else perturb for perturb in self.perturbs]))
        self.perturbs = torch.nn.ModuleList(perturbs)

    def to_dict(self, filt_type="default") -> dict:
        d = asdict_filt(self, filt_type=filt_type)
        if self.p < 1:
            d.update({"init_p": self.p})
        # As perturbs is ModuleList object, iterate it independly.
        perturbs = d.pop("perturbs")
        perturbs = [perturb.to_dict(filt_type=filt_type) for perturb in perturbs]
        cls = type(self)
        name = cls.name if hasattr(cls, "name") else cls.__name__
        return {"perturbs": perturbs, "name": name, **d}

    @abstractmethod
    def apply(self, **kwargs) -> ObjectDict:
        ...


class BaseSpeechAugmentConfig(metaclass=ABCMeta):
    """
    Base config of speechaugment configs.
    """

    CONFIGS = ObjectDict()

    def __init_subclass__(cls, **kwargs):
        name = cls.name if hasattr(cls, "name") else cls.__name__
        assert (
            name not in SignalPerturb.KNOWN_PERTURBS
        ), f"Failed register, config name:{name} is already known."
        BaseSpeechAugmentConfig.CONFIGS[name] = cls
        super().__init_subclass__(**kwargs)

    @property
    @abstractmethod
    def perturbs(self) -> Dict[str, Any]:
        ...

    @staticmethod
    def from_dict(data: dict) -> "BaseSpeechAugmentConfig":
        data = copy.deepcopy(data)
        cls = BaseSpeechAugmentConfig.get_register_class(data.pop("name"))
        return cls(**data)

    @staticmethod
    def get_register_class(name):
        assert name in BaseSpeechAugmentConfig.CONFIGS, f"Unknown config: {name}."
        return BaseSpeechAugmentConfig.CONFIGS[name]

    def to_dict(self, filt_type="default"):
        name = type(self).name if hasattr(type(self), "name") else type(self).__name__
        d = asdict_filt(self, filt_type=filt_type)
        return {**d, "name": name}


class NoiseSet:
    """
    Base class to construct noise dataset, which is available for signal perturb.
    e.g., read noises from noise dataset to inject into original audio.
    """

    DATASETS = ObjectDict()
    SUFFIX_MAP = ObjectDict()

    def __init_subclass__(cls, **kwargs):
        db_type = cls.db_type if hasattr(cls, "db_type") else cls.__name__
        assert (
            db_type not in NoiseSet.DATASETS
        ), f"Failed register, db_type:{db_type} exists."
        NoiseSet.DATASETS[db_type] = cls
        suffix = cls.suffix if hasattr(cls, "suffix") else f".{db_type.lower()}"
        assert (
            suffix not in NoiseSet.SUFFIX_MAP
        ), f"Failed register, name suffix:{suffix} exists."
        NoiseSet.SUFFIX_MAP[suffix] = db_type

        super().__init_subclass__(**kwargs)

    def sample(self, cnts: int = 1, resample: Optional[int] = None):
        """
        Random sample wav tensors from data_source.
        """
        raise NotImplementedError

    @staticmethod
    def from_dict(data: dict) -> "NoiseSet":
        db_type = data.pop("db_type", None)
        if db_type is None:
            try:
                suffix = Path(data["db_file"]).suffix
                db_type = NoiseSet.SUFFIX_MAP[suffix]
            except (KeyError, AttributeError):
                raise ValueError(
                    f'Auto infer db_type from db filename ({data["db_file"]}) error.'
                )
        class_name = NoiseSet.get_register_class(db_type)
        return class_name(**data)

    def to_dict(self, filt_type="default") -> dict:
        d = asdict_filt(self, filt_type=filt_type)
        db_type = (
            type(self).db_type
            if hasattr(type(self), "db_type")
            else type(self).__name__
        )
        return {**d, "db_type": db_type}

    @staticmethod
    def get_register_class(db_type):
        assert db_type in NoiseSet.DATASETS, f"Unknown noise dataset type: {db_type}"
        return NoiseSet.DATASETS[db_type]

    @classmethod
    @rich_exception_info
    def _load_utts(
        cls,
        wav_item,
        max_length=None,
        resample=16000,
    ) -> List["AudioClip"]:
        """
        As online resamping is ineffecient, we'd better do resampling
        before save noises to dataset.

        This function usually applied when creating noise dataset in subclasses.
        """
        if len(wav_item) == 2:
            utt, utt_path = wav_item
        elif len(wav_item) == 1:
            utt_path = wav_item
            utt = utt_path
        else:
            raise ValueError(
                "Expect 1 (utt_path) or 2 (utt, utt_path) lengths input items, but got len(wav_item)"
            )

        samples, sr = torchaudio.load(utt_path, normalize=False)
        bits = torch.iinfo(samples.dtype).bits

        if resample != sr:
            dtype = samples.dtype
            samples = samples / (1 << (bits - 1))
            samples = get_or_create_resampler(sr, resample)(samples)
            samples = (samples * (1 << (bits - 1))).to(dtype)
            sr = resample
        data = AudioClip(samples, sr, id=utt, bits_per_sample=bits)
        results = data.clip(max_length)

        return results

    @classmethod
    def create_db(
        cls,
        db_file: Union[str, Path],
        wave_items: Union[List[Tuple[str]], List[Tuple[str, str]]],
        mode: str = "w",
    ):
        """
        Create data base, to be implemented in derived classes.
        """
        raise NotImplementedError
