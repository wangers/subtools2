# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Leo 2023-03-01)

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torchaudio.transforms import Resample

from egrecho.data.datasets.audio.augments.dsp import (
    calculate_audio_energy,
    calculate_desired_audio_energy,
    fast_simulate_rir,
    fftconvolve1d,
    peak_normalize,
)
from egrecho.utils.common import ObjectDict, alt_none
from egrecho.utils.torch_utils import RandomValue, audio_collate_fn

from .base import MultiPerturb, NoiseSet, SignalPerturb, SinglePerturb
from .transforms import DropFreq, DropTime, Tempo


@dataclass(unsafe_hash=True)
class ChainPerturb(MultiPerturb):
    """
    This class sequencely apply augment subject to the given perturbs list.

    Arguments
    ---------
    init_p:
        Probability to be applied for the whole chain in the begining. Default 1.0
        While the unit probability is set in perturbs in the list.
    perturbs:
        Perturbs list, can be MultiPerturb or SinglePerturb.

    Example
    -------
    >>> perturbs=[
            {'name': 'speed', 'sample_rate': 16000, "factors":[0.9, 1.0, 1.1]},
            {'name': 'reverb'},
            {'name': 'wave_drop', 'init_p': 0.5},
        ]
    >>> signal = torch.sin(torch.arange(16000)).unsqueeze(0).unsqueeze(0)
    >>> chain = ChainPerturb(perturbs=perturbs)
    >>> signal = chain(signal).samples
    """

    name: ClassVar[str] = "chain_perturb"

    perturbs: List[Union[Dict, SignalPerturb]]

    init_p: Optional[float] = None

    def __post_init__(self):
        super().__post_init__()
        if self.init_p is not None:
            self.p = self.init_p
        if self.p < 1.0:
            self.init_p = self.p

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
        Arguments
        ---------
        samples : torch.Tensor
            Waveforms tensor with shape of `[batch, channel, time]` is compatible for all perturbs.
        lengths: torch.Tensor
            Valid lengths `[batch]`
        opts

        Returns
        -------
        output: ObjectDict
            A dict contains perturbed samples.
            i.e. `output_samples = output.samples` or `output_samples = output['samples']`.
        """
        inputs = ObjectDict(
            samples=samples,
            lengths=lengths,
            targets=targets,
            target_lengths=target_lengths,
            sample_rate=sample_rate,
            **kwargs,
        )
        for ptb in self.perturbs:
            inputs = ptb(**inputs)
        return inputs


@dataclass(unsafe_hash=True)
class RandomPerturb(MultiPerturb):
    """
    This class random select one perturb from the list.

    Arguments
    ---------
    init_p: float
        Probability to be applied. Default 1.0
        e.g., 0.6 means 60 percent of waves will flow through this part and left others unchanged.
    perturbs: List[Union[Dict, SignalPerturb]]
        Perturbs list, perturbs can be MultiPerturb or SinglePerturb.
    random_weight: Optional[List[float]]
        If None, uniformly select each perturbs, otherwise give them random weights.

    Example
    -------
    >>> perturbs=[
            {'name': 'wave_drop', 'init_p': 0.5},
            {'name': 'speed', 'sample_rate': 16000, "factors":[0.9, 1.0, 1.1]},
            {'name': 'reverb'}
            {'name': 'chain_perturb', 'perturbs':[{'name': 'reverb'}, {'name': 'wave_drop'}]}
            {'name': 'identity'}
        ]
    >>> random_weight = [1, 1, 1, 1, 4]  # in this case, clean and perturb will be equal.
    >>> signal = torch.sin(torch.arange(16000)).unsqueeze(0).unsqueeze(0)
    >>> random_aug = RandomPerturb(perturbs=perturbs, random_weight=random_weight)
    >>> signal = random_aug(signal).samples
    """

    name: ClassVar[str] = "random_perturb"

    perturbs: List[Union[Dict, SignalPerturb]]

    random_weight: Optional[List[float]] = None
    init_p: Optional[float] = None

    def __post_init__(self):
        super().__post_init__()
        if self.init_p is not None:
            self.p = self.init_p
        if self.p < 1.0:
            self.init_p = self.p

        self.random_weight = alt_none(self.random_weight, [1] * len(self.perturbs))

        if len(self.random_weight) != len(self.perturbs):
            raise ValueError(
                f"random_weight controls the prosibility of each "
                f"perturb in perturbs list when random select them. "
                f"Their lenght should be matched, but got "
                f"random_weight len: {len(self.random_weight)} "
                f"perturbs len: {len(self.perturbs)}. "
                f"You can set it None means uniform random choice."
            )
        self.random_weight_tensor = torch.tensor(self.random_weight, dtype=torch.float)

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
        Arguments
        ---------
        samples : torch.Tensor
            Waveforms tensor with shape of `[batch, channel, time]` is compatible for all perturbs.
        lengths: torch.Tensor
            Valid lengths `[batch]`
        opts

        Returns
        -------
        output: ObjectDict
            A dict contains perturbed samples.
            i.e. `output_samples = output.samples` or `output_samples = output['samples']`.
        """
        inputs = ObjectDict(
            samples=samples,
            lengths=lengths,
            targets=targets,
            target_lengths=target_lengths,
            sample_rate=sample_rate,
            **kwargs,
        )
        # random choice
        aug_idx = torch.multinomial(self.random_weight_tensor, 1)[0]
        for ptb_idx, ptb in enumerate(self.perturbs):
            if aug_idx == ptb_idx:
                inputs = ptb(**inputs)

        return inputs


@dataclass(unsafe_hash=True)
class Mixer(SinglePerturb):
    """
    Add noise.

    Arguments
    ---------
    noise_db_type: str
        Set to choose NoiseSet.
    noise_db: Path
        Noise location.
    noise_max_len: float
        Limit noise length.
    noise_filt_min: float
        Filter of noise.
    mix_num: Tuple
        A number will sample from this range to design the number of noises to be mixed,
        which simulates babble noise. e.g., (3, 7). Default (1, 1) means normal background noise case.
    noise_pad_mode: str
        Choice from ['repeat', 'retake']. When noise sample is shorter than reference wave, repeat it or load more noises.
    snr: Tuple
        Random choice snr for noises to be added in batch. e.g, (10, 20)
    init_p: float
        Probability to be applied.

    Example
    -------
    >>> Hdf5NoiseSet.creat_from_scp('test.h5', 'wav.scp')
    >>> mixer = Mixer(noise_db_type='hdf5', noise_db='test.h5', snr=(10, 20))
    >>> signal = torch.sin(torch.arange(16000)).unsqueeze(0).unsqueeze(0)
    >>> return_dct = mixer(signal)
    >>> perturb_signal = return_dct.samples
    """

    name: ClassVar[str] = "mix"

    noise_db: Union[str, Path]
    noise_db_type: Optional[str] = None
    noise_max_len: Optional[float] = None
    noise_filt_min: float = field(default=0.0, repr=False)

    mix_num: Tuple[int, int] = field(default_factory=lambda: (1, 1))
    snr: Tuple[float, float] = field(default_factory=lambda: (10, 20))
    noise_pad_mode: str = "retake"

    init_p: Optional[float] = None

    def __post_init__(self):
        super().__post_init__()
        if self.init_p is not None:
            self.p = self.init_p
        if self.p < 1.0:
            self.init_p = self.p

        self.db_conf = dict(
            db_type=self.noise_db_type,
            db_file=self.noise_db,
            max_len_cut=self.noise_max_len,
            filt_min=self.noise_filt_min,
        )
        self.snr_generater = RandomValue(start=self.snr[0], end=self.snr[-1])
        self.mix_num: RandomValue = RandomValue(
            start=self.mix_num[0], end=self.mix_num[-1]
        )
        if self.noise_pad_mode not in ["retake", "repeat"]:
            raise NotImplementedError

    def _load_noise(self, lengths, max_length):
        batch_size = len(lengths)
        if not hasattr(self, "noise_sampler"):
            self.noise_sampler = NoiseSet.from_dict(self.db_conf)
        mix_num = self.mix_num.sample_int().item()
        noise_sampler_cnt = batch_size
        sampler_lengths = lengths.tolist()
        if mix_num > batch_size:
            noise_sampler_cnt = mix_num
            sampler_lengths = sampler_lengths + [max(sampler_lengths)] * (
                mix_num - batch_size
            )

        noises = torch.zeros((noise_sampler_cnt, 1, max_length))

        for i in range(noise_sampler_cnt):
            noise, _ = self.noise_sampler.sample()
            # modify noises to cover whole valid signals.
            if noise.size(-1) < sampler_lengths[i]:
                if self.noise_pad_mode == "repeat":
                    repeat_num = math.ceil(sampler_lengths[i] / noise.size(-1))
                    repeat_nums = [1 for _ in range(noises.ndim - 1)] + [
                        repeat_num,
                    ]
                    noise = noise.repeat(repeat_nums)[..., : sampler_lengths[i]]
                # more compute cost but seems better.
                elif self.noise_pad_mode == "retake":
                    pieces = [noise]
                    missing_num_samples = sampler_lengths[i] - noise.size(-1)
                    while missing_num_samples > 0:
                        piece, _ = self.noise_sampler.sample()
                        missing_num_samples -= piece.size(-1)
                        pieces.append(piece)
                    noise = torch.cat(
                        [
                            piece
                            / (calculate_audio_energy(piece, rms_energy=True)).clip(
                                min=1e-8
                            )
                            for piece in pieces
                        ],
                        dim=1,
                    )[..., : sampler_lengths[i]]
                else:
                    raise NotImplementedError
            else:
                start = random.randint(0, noise.size(-1) - sampler_lengths[i])
                noise = noise[..., start : start + sampler_lengths[i]]
            noises[i][..., : sampler_lengths[i]] += noise / calculate_audio_energy(
                noise, rms_energy=True
            ).clip(min=1e-8)

        noises = noises.to(lengths.device)
        if mix_num > 1:
            babble_noises = noises.clone()
            for i in range(1, mix_num):
                babble_noises += noises.roll((i,), dims=0)
            noises = babble_noises[:batch_size]

            mask = torch.arange(0, noises.size(-1), device=noises.device)
            noises = noises.masked_fill(mask >= lengths[..., None, None], 0.0)
            del mask
            noises = noises / calculate_audio_energy(
                noises, lengths=lengths, rms_energy=True
            ).clip(min=1e-8)

        return noises

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
        Arguments
        ---------
        samples: tensor `[batch, channel, time]`
        lengths: tensor `batch` or None
            Valid lengths.

        Returns
        -------
        output: ObjectDict
            A dict contains perturbed samples.
        """
        assert (
            samples.ndim == 3
        ), f"Expect 3 dim input tensor (batch, channle, time), but got {samples.ndim}"
        batch_size, channel, max_length = samples.shape
        noise_lens = alt_none(
            lengths,
            torch.ones(batch_size, dtype=torch.int32, device=samples.device)
            * max_length,
        )
        noise_batch = self._load_noise(noise_lens, max_length)

        ref_rms = calculate_audio_energy(samples, lengths, rms_energy=True)
        snr = self.snr_generater.sample(
            shape=(batch_size, channel, 1), device=samples.device
        )
        gain = calculate_desired_audio_energy(ref_rms, snr, rms_energy=True)

        samples = samples + gain * noise_batch

        # normalize out range value
        samples = peak_normalize(samples)
        return ObjectDict(
            samples=samples,
            lengths=lengths,
            targets=targets,
            target_lengths=target_lengths,
            sample_rate=sample_rate,
            **kwargs,
        )


@dataclass(unsafe_hash=True)
class ResponseImpulse(SinglePerturb):
    """
    Reverberation effect by convolving with a room impulse response.
    As convolve brings delays, here automatically compensate the delay through offset results.
    i.e., output length will be equal to the input length.

    Arguments
    ---------
    noise_db: Path
        If not None, load rirs from this Noise location.
        Otherwise apply on-the-fly rir simulation.
    noise_db_type: str
        Set to choose NoiseSet if load rirs from noisedataset.
    rir_fix_len:
        If set True, loaded rir kernel longer than signal will slice to the wave length from head.
    sim_rir_conf: dict
        The config for simulate rir when noise_db is None.
    init_p: float
        Probability to be applied.

    Example
    -------
    >>> Hdf5NoiseSet.creat_from_scp('test.h5', 'wav.scp')
    >>> rir_conv = ResponseImpulse(noise_db_type='hdf5', noise_db='test.h5')
    >>> signal = torch.sin(torch.arange(16000)).unsqueeze(0).unsqueeze(0)
    >>> return_dct = rir_conv(signal)
    >>> perturb_signal = return_dct.samples
    """

    name: ClassVar[str] = "reverb"

    noise_db: Optional[Union[str, Path]] = None
    noise_db_type: Optional[str] = None

    sim_prob: float = 0.0
    sim_rir_conf: Optional[dict] = field(
        default_factory=lambda: dict(sample_rate=16000, max_D=36, max_R=1.6)
    )
    rir_fix_len: bool = True

    init_p: Optional[float] = None

    def __post_init__(self):
        super().__post_init__()
        if self.init_p is not None:
            self.p = self.init_p
        if self.p < 1.0:
            self.init_p = self.p

        assert 0 <= self.sim_prob <= 1
        self.db_conf = dict(
            db_type=self.noise_db_type,
            db_file=self.noise_db,
        )

    def _load_rir(self, lengths):
        batch_size = len(lengths)
        if not hasattr(self, "noise_sampler"):
            self.noise_sampler = NoiseSet.from_dict(self.db_conf)
        noise_sampler_cnt = batch_size

        noises = []
        for i in range(noise_sampler_cnt):
            noise, rate = self.noise_sampler.sample()

            # when rir wav lengths is longer
            if noise.size(-1) > lengths[i] and self.rir_fix_len:
                direct_index = noise.abs().argmax(axis=1)[0].item()
                if direct_index > lengths[i]:
                    end = min(direct_index + lengths[i] // 2, lengths[i])
                    noise = noise[
                        ...,
                        end - lengths[i] : end,
                    ]
                else:
                    noise = noise[..., 0 : lengths[i] + 1]
            noises.append(noise)

        noises, _ = audio_collate_fn(noises)

        return noises.to(lengths.device)

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
        Arguments
        ---------
        samples: tensor `[batch, channel, time]`.
        lengths: int
            Valid lengths `[batch]`

        Returns
        -------
        Tensor: Shape  of `[batch, channel, time]`.
        """
        assert (
            samples.ndim == 3
        ), f"Expect 3 dim input tensor (batch, channle, time), but got {samples.ndim}"
        batch_size, channel = samples.shape[:-1]
        true_lengths = alt_none(
            lengths,
            torch.ones(batch_size, dtype=torch.int32, device=samples.device)
            * samples.shape[-1],
        )
        if torch.rand(1) >= self.sim_prob:
            rir_ = self._load_rir(true_lengths)
        else:
            rir_, _ = fast_simulate_rir(batch=batch_size, **self.sim_rir_conf)
            rir_ = rir_.to(samples.device)

        assert (
            rir_.shape[1] == 1
        ), f"Now just support momo rir signal, but got channel dim {rir_.shape[1]}."
        # Record a delay to offset the reverbed audio. [B,]
        delays = rir_.abs().argmax(dim=2, keepdim=False)[:, 0]

        origin_energy = calculate_audio_energy(samples, lengths)

        conv_samples = fftconvolve1d(samples, rir_.expand(-1, channel, -1), mode="full")
        samples = torch.stack(
            [
                conv_sample[:, delay : delay + true_length]
                for conv_sample, true_length, delay in zip(
                    conv_samples, true_lengths, delays
                )
            ],
            dim=0,
        )
        if lengths is not None:
            mask = torch.arange(0, samples.shape[-1], device=samples.device)
            samples = samples.masked_fill(mask >= lengths[..., None, None], 0.0)
        new_energy = calculate_audio_energy(samples, lengths).clip(min=1e-8)

        samples = samples * torch.sqrt(origin_energy / new_energy)

        return ObjectDict(
            samples=samples,
            lengths=lengths,
            targets=targets,
            target_lengths=target_lengths,
            sample_rate=sample_rate,
            **kwargs,
        )


def _source_target_sample_rate(orig_freq: int, speed: float) -> Tuple[int, int]:
    source_sample_rate = int(speed * orig_freq)
    target_sample_rate = int(orig_freq)
    gcd = math.gcd(source_sample_rate, target_sample_rate)
    return source_sample_rate // gcd, target_sample_rate // gcd


@dataclass(unsafe_hash=True)
class SpeedPerturb(SinglePerturb):
    """
    Speed perturbation based on resampling effect or sox_tempo.
    Resampling changes pitch while sox_tempo preserves.
    For sv task, resampling waves could be regarded as another speaker, i.e. affects spk_id.

    NOTE: applies the same perturb to examples from a batch.
    For fix chunk training in SV task (the most common situation), we'd better to resize the shape after
    speed perturbation to preserve the length of input.
    Particularly, the 'resample' method may change the pitch of wavs. The pith_shift factor attached
    with the length perturbation leads to confusion during training.
    So applys speed perturbation with `resize_shape=False` before getting fix chunks
    or set `resize_shape=True` otherwise.

    Arguments
    ---------
    sample_rate: int
        Original sample rate of sinals to be perturbed.
    method: str
        Choice from `['tempo', 'resample']`.
    init_p: float
        Probability to be applied.
    resize_shape: bool
        Truncate/zero-pad the result signal.
    record_resample_factor: bool
        If True, it'll record the factor of speed perturbation,
        which is used to affect spk_id when the method is `resample`.
    factors : Sequence[float]
        e.g. [0.95, 1, 1.05], larger -> faster.

    Example
    -------
    >>> signal = torch.sin(torch.arange(16000)).unsqueeze(0).unsqueeze(0)
    >>> factors = [0.9, 1.2]
    >>> sp = SpeedPerturb(sample_rate=16000, factors=factors, record_resample_factor=True)
    >>> return_dct = sp(signal)
    >>> perturb_signal, resample_factor = return_dct.samples, return_dct.resample_factor
    >>> signal.shape, perturb_signal.shape, resample_factor
    (torch.Size([1, 1, 16000]), torch.Size([1, 1, 13334]), 1.2)
    """

    name: ClassVar[str] = "speed"

    sample_rate: int
    method: str = "resample"
    resize_shape: bool = False
    record_resample_factor: bool = False
    factors: Sequence[float] = field(default_factory=lambda: [0.9, 1.0, 1.1])
    init_p: Optional[float] = None

    def __post_init__(self):
        assert self.method in [
            "tempo",
            "resample",
        ], f"Unsupport speed perturb method: {self.method}."
        super().__post_init__()
        if self.init_p is not None:
            self.p = self.init_p
        if self.p < 1.0:
            self.init_p = self.p

        self.source_target_sample_rate = [
            _source_target_sample_rate(self.sample_rate, factor)
            for factor in self.factors
        ]

        if self.method == "resample":
            self.speeders = torch.nn.ModuleList(
                [
                    Resample(orig_freq=orig_freq, new_freq=new_freq)
                    for orig_freq, new_freq in self.source_target_sample_rate
                ]
            )
        elif self.method == "tempo":
            self.speeders = torch.nn.ModuleList(
                [
                    Tempo(sample_rate=self.sample_rate, factor=factor)
                    for factor in self.factors
                ]
            )
        else:
            raise NotImplementedError

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
        Arguments
        ---------
        samples: tensor `[batch, channel, time]`.
        lengths: int
            Valid lengths `[batch]`

        Returns
        -------
        Tensor: Shape  of `[batch, channel, time]`.
        """
        assert (
            samples.ndim == 3
        ), f"input dim should be 3 with [batch, channel, time], but got {samples.ndim}."
        idx = int(torch.randint(len(self.speeders), ()))
        b, c, t = samples.shape

        for speeder_idx, speeder in enumerate(self.speeders):
            if idx == speeder_idx:
                samples = speeder(samples)

        if self.resize_shape:
            # Managing speed change
            if samples.shape[-1] > t:
                samples = samples[..., :t]
            else:
                zero_sig = torch.zeros(b, c, t, device=samples.device)
                zero_sig[..., : samples.shape[-1]] = samples
                samples = zero_sig
        else:
            # Get valid lengths
            if lengths:
                if self.method == "resample":
                    # scale: length * (target_sr/source_sr)
                    lengths = torch.ceil(
                        lengths
                        * self.source_target_sample_rate[idx][1]
                        / self.source_target_sample_rate[idx][0]
                    ).to(lengths.dtype)
                else:
                    lengths = torch.floor(lengths * (1 / self.factors[idx])).to(
                        lengths.dtype
                    )

        out_put = ObjectDict(
            samples=samples,
            lengths=lengths,
            targets=targets,
            target_lengths=target_lengths,
            sample_rate=sample_rate,
            **kwargs,
        )
        # record sp factor if needed.
        if self.method == "resample" and self.record_resample_factor:
            out_put.resample_factor = self.factors[idx]
        return out_put


@dataclass(unsafe_hash=True)
class WaveDrop(SinglePerturb):
    """
    Combine of `DropFreq` and `DropTime`

    1. Drop chunks of the audio
    2. Drop frequency bands (with band-drop filters)

    Arguments
    ---------
    init_p: float
        Probability to be applied.
    sample_rate: int
        Sample rate of inputs, influences the drop_length.
    drop_time_count : [int, int]
        Random selected a number `n` in range to drop chunks to zero for `n` times.
    drop_freq_count: [int, int]
        Random selected a number `n` in range to drop frequencies `n` times.
    drop_time_length : [float, float]
        Random selected a length (in Seconds) in this range to set signal samples to zero.

    Example
    -------
    >>> inputs = torch.randn([10, 16000])
    >>> wave_drop = WaveDrop()
    >>> signals = wave_drop(inputs).samples
    """

    name: ClassVar[str] = "wave_drop"

    sample_rate: int = 16000
    drop_time_count: Sequence[int] = field(default_factory=lambda: [0, 4])
    drop_freq_count: Sequence[int] = field(default_factory=lambda: [0, 3])
    drop_time_length: Sequence[int] = field(default_factory=lambda: [0.065, 0.125])
    init_p: Optional[float] = None

    def __post_init__(self):
        super().__post_init__()
        if self.init_p is not None:
            self.p = self.init_p
        if self.p < 1.0:
            self.init_p = self.p

        self.drop_freq = DropFreq(
            drop_count=self.drop_freq_count,
        )
        self.drop_time = DropTime(
            drop_count=self.drop_time_count,
            drop_length=[
                int(self.drop_time_length[0] * self.sample_rate),
                int(self.drop_time_length[-1] * self.sample_rate),
            ],
        )

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
        Arguments
        ---------
        samples: tensor `[..., time]`.

        Returns
        -------
        Tensor: Shape  of `[..., time]`.
        """
        samples = self.drop_freq(samples)
        samples = self.drop_time(samples)
        return ObjectDict(
            samples=samples,
            lengths=lengths,
            targets=targets,
            target_lengths=target_lengths,
            sample_rate=sample_rate,
            **kwargs,
        )


@dataclass(unsafe_hash=True)
class Identity(SinglePerturb):
    """
    Identity link, this class just make augment pipline more flexible.
    """

    name: ClassVar[str] = "identity"

    def __post_init__(self):
        super().__post_init__()

        # Actually waveforms will not flow into here.
        self.p = 0

    def apply(
        self,
        samples: torch.Tensor = None,
        lengths: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        sample_rate: Optional[int] = None,
        **kwargs,
    ) -> ObjectDict:
        return ObjectDict(
            samples=samples,
            lengths=lengths,
            targets=targets,
            target_lengths=target_lengths,
            sample_rate=sample_rate,
            **kwargs,
        )
