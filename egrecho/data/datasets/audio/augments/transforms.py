# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Leo 2023-01-11)

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import torch
import torchaudio

from egrecho.data.datasets.audio.augments.dsp import fftconvolve1d, notch_filter
from egrecho.utils.torch_utils import RandomValue


@dataclass(unsafe_hash=True)
class DropFreq(torch.nn.Module):
    """
    This class drops random frequencies from the signal.

    Implementation based on speechbrain.
    https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/processing/speech_augmentation.py

    The purpose of this class is to teach models to learn to rely on all parts
    of the signal, not just a few frequency bands.

    Note: Each samples `batch*channels` drop the same frequencies.

    Arguments
    ---------
    drop_count: Sequence[int, int]
        Random selected a number `n` in range to drop frequencies `n` times.
    drop_freq: Sequence[float, float]
        The range of frequencies that can be dropped,
        as a fraction of the sampling rate / 2.
    drop_width: float
        The width of the frequency band to drop, as
        a fraction of the sampling_rate / 2.
    use_fft: bool
        If True, use FFTs instead of convolve when applying a notch filter to drop frequency.

    Example
    -------
    >>> import torchaudio
    >>> dropper = DropFreq()
    >>> signal, _ = torchaudio.load('samples/audio_samples/example1.wav')
    >>> dropped_signal = dropper(signal)
    """

    use_fft: bool = True
    drop_count: Sequence[int] = field(default_factory=lambda: [0, 3])
    drop_freq: Sequence[float] = field(default_factory=lambda: [1e-14, 1.0], repr=False)
    drop_width: float = field(default=0.05, repr=False)

    def __post_init__(self):
        super().__init__()
        self.drop_count_sampler = RandomValue(
            start=self.drop_count[0], end=self.drop_count[-1]
        )
        self.drop_freq_sampler = RandomValue(
            start=self.drop_freq[0], end=self.drop_freq[-1]
        )

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Arguments
        ---------
        samples: Waveform tensor
             Shape should be `[..., time]`.

        Returns
        -------
        Tensor: Shape of `[..., time]`
        """
        output_shape = samples.shape[:-1] + (-1,)

        # Pick number of frequencies to drop
        drop_count = self.drop_count_sampler.sample_int(shape=(1,))
        # Pick a frequency to drop
        drop_frequency = self.drop_freq_sampler.sample(drop_count)

        # Filter parameters
        filter_length = 101
        pad = filter_length // 2

        # Start with delta function
        drop_filter = torch.zeros(1, 1, filter_length, device=samples.device)
        drop_filter[..., pad] = 1

        # Subtract each frequency
        for frequency in drop_frequency:
            notch_kernel = notch_filter(
                frequency,
                filter_length,
                self.drop_width,
            ).to(samples.device)
            drop_filter = torch.nn.functional.conv1d(
                input=drop_filter, weight=notch_kernel, padding=pad
            )

        # Apply filter
        if not self.use_fft:
            num_samples = torch.tensor(samples.shape[:-1]).prod()
            drop_filter = drop_filter.expand(int(num_samples), 1, -1)
            samples = torch.nn.functional.conv1d(
                input=samples.reshape(int(num_samples), -1),
                weight=drop_filter,
                groups=num_samples,
                padding=pad,
            )
        else:
            samples = fftconvolve1d(samples, drop_filter, mode="same")

        samples = samples.reshape(output_shape)
        return samples


@dataclass(unsafe_hash=True)
class DropTime(torch.nn.Module):
    """
    This class drops portions of the input signal.

    Using `DropTime` as an augmentation strategy helps a models learn to rely
    on all parts of the signal, since it can't expect a given part to be
    present.

    Note: Each samples `batch * channels` drop the same times (drop_cout) but different lengths (drop_length).

    Arguments
    ---------
    drop_count: [int, int]
        Random selected a number `n` in range to drop chunks to zero for `n` times.
    drop_length: [int, int]
        Random selected a length in this range to set signal samples to zero.

    Example
    -------
    >>> inputs = torch.arange(10).view(2,1,5)
    >>> drop_time = DropTime(drop_count=[1, 2], drop_length=[2, 2])
    >>> drop_time(inputs)
    tensor([[[0, 0, 0, 0, 4]],
           [[0, 0, 7, 8, 9]]])
    """

    drop_count: Sequence[int] = field(default_factory=lambda: [0, 4])
    drop_length: Sequence[int] = field(default_factory=lambda: [1000, 2000])

    def __post_init__(self):
        super().__init__()
        assert self.drop_length[0] >= 0
        self.drop_count_sampler = RandomValue(
            start=self.drop_count[0], end=self.drop_count[-1]
        )
        self.drop_length_sampler = RandomValue(
            start=self.drop_length[0], end=self.drop_length[-1]
        )

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Drop chunks along time axis.

        Arguments
        ---------
        samples : Tensor `[..., time]`.

        Returns
        -------
        Tensor: Shape  of `[..., time]`.
        """

        # Pick a number of times to drop
        drop_times = self.drop_count_sampler.sample_int(shape=(1,))
        if drop_times <= 0:
            return samples
        device = samples.device

        # Pick drop lengths.
        drop_lengths = (
            self.drop_length_sampler.sample_int((drop_times,) + samples.shape[:-1])
        ).to(device)
        start = torch.rand((drop_times,) + samples.shape[:-1], device=device) * (
            samples.shape[-1] - drop_lengths
        )

        # Creat mask for broadcasting
        mask_start = start.long()[..., None]  # [drop_times, ..., time]
        mask_end = (start.long() + drop_lengths.long())[
            ..., None
        ]  # [drop_times, ..., time]
        mask = torch.arange(0, samples.shape[-1], device=device)

        # Iterate drop_times to set mask
        for i in range(drop_times):
            samples = samples.masked_fill(
                (mask >= mask_start[i]) & (mask < mask_end[i]), 0.0
            )
        return samples


@dataclass(unsafe_hash=True)
class Tempo(torch.nn.Module):
    """
    Apply `sox tempo` effect to given Tensor.
    Compared to resample-based speed perturbation, tempo preserves pitch.

    Note: Works with batch by processing each example in a loop.
    Input should be `[batch, channels, time]`

    Arguments
    ---------
    factors: float
    """

    factor: float
    sample_rate: int

    def __post_init__(self):
        super().__init__()

    def forward(
        self,
        samples: torch.Tensor,
        sample_rate: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Arguments
        ---------
        samples: tensor `[batch, channel, time]`.
        sample_rate: int
            sample_rate of input signals.

        Returns
        -------
        Tensor: Shape  of `[batch, channel, time]`.
        """

        assert (
            samples.ndim == 3
        ), f"input dim should be 3 with [batch, channel, time], but got {samples.ndim}."
        sample_rate = self.sample_rate or sample_rate
        if sample_rate is None:
            raise RuntimeError("sample_rate is required")
        device = samples.device
        if self.factor == 1.0:
            return samples
        # Only works on CPU Tensors.
        samples = samples.to("cpu")

        outs = []
        # Iterates sounds.
        for sample in samples:
            sample, _ = torchaudio.sox_effects.apply_effects_tensor(
                sample, sample_rate, [["tempo", str(self.factor)]]
            )
            outs.append(sample)

        samples = torch.stack(outs).to(device)
        return samples


_precompiled_resamplers: Dict[Tuple[int, int], torch.nn.Module] = {}


def get_or_create_resampler(
    source_sampling_rate: int, target_sampling_rate: int
) -> torch.nn.Module:
    global _precompiled_resamplers

    tpl = (source_sampling_rate, target_sampling_rate)
    if tpl not in _precompiled_resamplers:
        _precompiled_resamplers[tpl] = torchaudio.transforms.Resample(
            source_sampling_rate, target_sampling_rate
        )
    return _precompiled_resamplers[tpl]
