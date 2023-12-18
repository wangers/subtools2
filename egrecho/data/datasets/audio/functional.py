# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-07)

import math
import random
from typing import Any, Optional, Union, cast

import torch
from torch import Tensor

from egrecho.data import processors
from egrecho.data.datasets.constants import (
    AUDIO_COLUMN,
    OFFLINE_FEAT_COLUMN,
    SAMPLE_RATE_COLUMN,
    SPEAKER_COLUMN,
)
from egrecho.utils.logging import get_logger
from egrecho.utils.misc import ConfigurationException, NoneDataException
from egrecho.utils.torch_utils import audio_collate_fn, tensor_has_nan

from .augments import ASVSpeechAgugmentConfig, SpeechAgugment, SpeedPerturb
from .augments.dsp import de_silence
from .augments.transforms import get_or_create_resampler

logger = get_logger(__name__)


def de_sil(data, win_len=0.1, min_eng=50, retry_times=1, force_output=True):
    """Trim silence.

    Args:
        data: Iterable[{audio, sample_rate, ...}]
        resample_rate: target resample rate

    Returns:
        Iterable[{audio, sample_rate, ...}]
    """
    for sample in data:
        waveform = sample[AUDIO_COLUMN]
        sr = sample[SAMPLE_RATE_COLUMN]
        cache_wave, cache_len = de_silence(
            waveform, sr=sr, win_len=win_len, min_eng=min_eng
        )
        retry_left = retry_times
        while retry_left and cache_len == 0:
            min_eng /= 2
            cache_wave, _ = de_silence(
                waveform, sr=sr, win_len=win_len, min_eng=min_eng
            )
            retry_left -= 1
        if force_output and cache_len == 0:
            cache_wave = waveform
        sample[AUDIO_COLUMN] = cache_wave
        del waveform
        yield sample


def select_channel(data, channle_id: int = 0):
    """take one channel of audio,

    Args:
        data: Iterable[{audio, ...}]
        channle_id: the chosen channel.

    Returns:
        Iterable[{audio, ...}]
    """
    for sample in data:
        waveform: Tensor = sample[AUDIO_COLUMN]

        if waveform.shape[-2] - 1 < channle_id:
            raise ValueError(f"Audio: {sample} lack channle {channle_id}.")
        elif waveform.shape[-2] - 1 == channle_id == 0:
            yield sample
        else:
            sample[AUDIO_COLUMN] = waveform[..., channle_id : channle_id + 1, :]
            yield sample


def filter(data, max_length=15.0, min_length=0.1, truncate=True):
    """
    Filter.

    Args::
        data: Iterable[{audio, sample_rate, ...}]
        max_length: drop utterance which is greater than max_length(s)
        min_length: drop utterance which is less than min_length(s).

    Returns:
        Iterable[{audio, sample_rate, ...}]
    """

    for sample in data:
        waveform: Tensor = sample[AUDIO_COLUMN]
        sample_rate = sample[SAMPLE_RATE_COLUMN]
        duration = waveform.size(1) / sample_rate
        if duration < min_length:
            continue
        if duration > max_length:
            if truncate:
                sample[AUDIO_COLUMN] = apply_random_chunk(
                    waveform, max_length * sample_rate
                )
            else:
                continue

        yield sample


def resample(data, resample_rate=16000):
    """Resample data.

    Args:
        data: Iterable[{audio, sample_rate, ...}]
        resample_rate: target resample rate

    Returns:
        Iterable[{audio, sample_rate, ...}]
    """
    for sample in data:
        sample_rate = sample[SAMPLE_RATE_COLUMN]
        if sample_rate != resample_rate:
            waveform = sample[AUDIO_COLUMN]
            sample[SAMPLE_RATE_COLUMN] = resample_rate
            sample[AUDIO_COLUMN] = get_or_create_resampler(
                source_sampling_rate=sample_rate, target_sampling_rate=resample_rate
            )(waveform)
        yield sample


def _affix_speaker(speaker: str, effect: Any, prefix=""):
    prefix = f"_{prefix}" if prefix else ""
    return f"{speaker}{prefix}_{effect}"


class PreSpeedPerturb(object):
    """Resample-based speed perturb.

    Applys speed perturb and records speed factor to affect spk-id.

    Args:
        sample_rate: int
            Original sample rate of sinals to be perturbed.
        affix_speaker: bool
            If True, it'll affix the speaker colunm according to the resample factor applied.
        factors : Sequence[float]
            e.g. [0.95, 1, 1.05], larger -> faster.
    """

    def __init__(
        self,
        sample_rate=16000,
        factors=(0.9, 1.0, 1.1),
        affix_speaker=True,
    ):
        super().__init__()
        if affix_speaker and (1.0 not in tuple(factors)):
            raise ConfigurationException(
                "It seems speed perturb will affect speaker, but failed to "
                f"get a factor of {1.0} to keep a version of original speaker. "
                f"Reset factors to like {0.9, 1.0, 1.1}."
            )
        self.speeder = SpeedPerturb(
            sample_rate,
            factors=factors,
            record_resample_factor=affix_speaker,
        )

    def __call__(self, data):
        r"""Speed perturb.

        Input audio tensor has shape of `C * T`, while `speeder` module need 3-dim inputs,
        first adds 0-dimension and removes it at last.

        Args:
            data: Iterable[{audio, sample_rate, ...}]
        Returns:
            Iterable[{audio, sample_rate, ...}]
        """
        for sample in data:
            waveform: Tensor = sample[AUDIO_COLUMN]
            assert waveform.ndim == 2
            if sample[SAMPLE_RATE_COLUMN] != self.speeder.sample_rate:
                raise ValueError(
                    "sample_rate should aligns to sample_rate of speeder: "
                    f"({self.speeder.sample_rate}), but got {sample}."
                )
            aug_output = self.speeder(samples=waveform.unsqueeze(0))

            resample_factor = aug_output.get("resample_factor")
            if (
                resample_factor is not None
                and resample_factor != 1
                and sample.get(SPEAKER_COLUMN) is not None
            ):
                speaker = sample[SPEAKER_COLUMN]
                speaker = _affix_speaker(speaker, resample_factor, prefix="sp")
                sample[SPEAKER_COLUMN] = speaker

            aug_waveform = cast(Tensor, aug_output.samples)
            sample[AUDIO_COLUMN] = aug_waveform.squeeze(0)
            yield sample

    def train(self, mode: bool = True):
        r"""Sets training mode."""
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        self.speeder.train(mode)
        return self

    def eval(self):
        r"""Sets evaluation mode."""
        return self.train(False)


def random_chunk(
    data,
    chunk_len: int,
    data_type: str = "raw",
    train_mode: bool = True,
    retry: Union[int, bool] = 0,
    force_retry: bool = True,
    retry_ampth: float = 2e-4,
):
    """fixed-chunk

    Args:
        data:
            Iterable[{audio, ...}]
        chunk_len:
            Random chunk length
        train_mode:
            If False, always return the middle chunk.
        retry:
            This is only use for the case of input is audio signal with train mode.
            If > 0, will retry random chunk to avoid it with very small energy. if it is still invalid after that,
            returns the last retried chunk or filter it according to param `force_retry`.
        force_retry:
            Force output when failed retrying.
        retry_ampth:
            The mean value of chunk lower then this threashould is treated as invalid when applying retry.
    Returns:
        Iterable[{audio, ...}]
    """
    if data_type == "offline_feat":
        for sample in data:
            feature: Tensor = sample[OFFLINE_FEAT_COLUMN]
            sample[OFFLINE_FEAT_COLUMN] = apply_random_chunk(
                feature.transpose(-2, -1), chunk_len, train_mode
            ).transpose(-2, -1)
            yield sample
    else:
        for sample in data:
            waveform: Tensor = sample[AUDIO_COLUMN]
            chunk = apply_random_chunk(waveform, chunk_len, train_mode)
            retry_left = int(retry)

            if train_mode and retry_left > 0:
                length = waveform.shape[-1]
                while (
                    retry_left
                    and length > chunk_len
                    and not is_valid_chunk(chunk, retry_ampth)
                ):
                    chunk = apply_random_chunk(waveform, chunk_len, train_mode)
                    retry_left -= 1
                if is_valid_chunk(chunk, retry_ampth) or force_retry:
                    sample[AUDIO_COLUMN] = chunk
                else:
                    continue
            else:
                sample[AUDIO_COLUMN] = chunk
            yield sample


def apply_random_chunk(data: Tensor, chunk_len: int, train_mode: bool = True):
    """fixed-chunk on the last dimention.

    Args:
        data:
            wav or feature.
        chunk_len:
            output length.
    Returns:
        Tensor
    """
    data_len = data.shape[-1]
    chunk_len = int(chunk_len)
    # random chunk
    if data_len > chunk_len:
        if train_mode:
            start = random.randint(0, data_len - chunk_len)
        else:
            start = (data_len - chunk_len) // 2
        data = data[..., start : start + chunk_len]
        data = data.clone()
    else:
        repeat_num = math.ceil(chunk_len / data_len)
        repeat_shape = [
            1,
        ] * (data.ndim - 1)
        repeat_shape += [
            repeat_num,
        ]
        data = data.repeat(repeat_shape)[..., :chunk_len]
    return data


def is_valid_chunk(data: Tensor, mean_th: float = 5e-4):
    if torch.mean(torch.abs(data[0])) < mean_th:
        return False
    return True


class SpeechAugPipline(object):
    """applys various speechaug.

    first batch audio to a `batch_size`, after speech augment unbatch to element-wise.
    see `egrecho.data.egs.audio.augments.SpeechAgugment`. support batch-wise processing.

    Args:
        db_dir: str
            db_noise base dir.
        batch_size: bool
        speech_aug_path : str
            json/yaml file of config, if None, use default `ASVSpeechAgugmentConfig`,
            which needs a `db_dir`.
        **kwds:
            override args.
    """

    def __init__(
        self,
        db_dir: Optional[str] = None,
        batch_size: int = 1,
        speech_aug_path: Optional[str] = None,
        **kwds,
    ):
        super().__init__()
        self.batch_size = batch_size
        if speech_aug_path is not None:
            from egrecho.utils.io import SerializationFn

            speech_aug_dict = SerializationFn.load_file(speech_aug_path)
            if db_dir is not None:
                kwds["db_dir"] = db_dir
            speech_aug_dict.update(kwds)
            speech_aug_config = ASVSpeechAgugmentConfig.from_dict(speech_aug_dict)
        else:
            speech_aug_config = ASVSpeechAgugmentConfig(db_dir=db_dir, **kwds)
        self.speechaug = SpeechAgugment(speech_aug_config)

    def __call__(self, data, ignore_lengths: bool = False):
        """speechaug.
        Args:
            data: Iterable[{audio, sample_rate, ...}]
        Returns:
            Iterable[{audio, sample_rate, ...}]
        """
        for batch_sample in processors.batch(data, self.batch_size):
            waveforms = [sample[AUDIO_COLUMN] for sample in batch_sample]
            waveforms, lengths = audio_collate_fn(waveforms)
            if ignore_lengths:
                lengths = None
            aug_output = self.speechaug(waveforms, lengths)  # output is dict
            aug_waveforms = cast(Tensor, aug_output.samples)
            aug_lengths = cast(Tensor, aug_output.lengths)
            for batch_idx, aug_wav in enumerate(aug_waveforms):
                if tensor_has_nan(aug_wav):
                    logger.warning(
                        f"speechaug:  has None. Trying to use original data."
                    )
                    # if not tensor_has_nan(batch_sample[batch_idx][AUDIO_COLUMN]):
                    #     pass
                    # else:
                    #     raise NoneDataException("Got None data before speechaug.")
                    continue
                else:
                    if not ignore_lengths:
                        aug_wav = aug_wav[..., : aug_lengths[batch_idx]]
                    batch_sample[batch_idx][AUDIO_COLUMN] = aug_wav
                yield batch_sample[batch_idx]

    def train(self, mode: bool = True):
        r"""Sets training mode."""
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        self.speechaug.train(mode)
        return self

    def eval(self):
        r"""Sets evaluation mode."""
        return self.train(False)

    def __repr__(self) -> str:
        return f"{repr(self.speechaug)}\n(batch_size): {self.batch_size}"
