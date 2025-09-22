# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Leo 2023-01-11)
"""Low level signal processing utilities."""

import io
import math
import warnings
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.fft import irfft, rfft
from torch.nn.utils.rnn import pad_sequence
from torchaudio.functional import highpass_biquad

_NEXT_FAST_LEN = {}


def next_fast_len(size):
    """
    Returns the next largest number ``n >= size`` whose prime factors are all
    2, 3, or 5. These sizes are efficient for fast fourier transforms.
    Equivalent to :func:`scipy.fftpack.next_fast_len`.

    Note: This function was originally copied from
    https://github.com/pyro-ppl/pyro
    repository, where the license was Apache 2.0. Any modifications to the original code can be
    found at https://github.com/asteroid-team/torch-audiomentations/commits
    :param int size: A positive number.
    :returns: A possibly larger number.
    :rtype int:
    """
    try:
        return _NEXT_FAST_LEN[size]
    except KeyError:
        pass

    assert isinstance(size, int) and size > 0
    next_size = size
    while True:
        remaining = next_size
        for n in (2, 3, 5):
            while remaining % n == 0:
                remaining //= n
        if remaining == 1:
            _NEXT_FAST_LEN[size] = next_size
            return next_size
        next_size += 1


# Implementation based on torch-audiomentations:
# https://github.com/asteroid-team/torch-audiomentations/blob/master/torch_audiomentations/utils/convolution.py
def fftconvolve1d(
    waveform: Tensor,
    kernel: Tensor,
    mode: str = "full",
):
    """
    Computes the 1-d convolution of signal by kernel along their last dimension using FFTs.

    For inputs with large size of the last dimension, it is more effecient.
    Signal and kernel should have the same dim or can be broadcated.

    Arguments
    ---------
    waveforms:
        The waveforms of shape `[...,time]`.
    kernel:
        A convolution kernel.
    mode:
        One of: `['full', 'valid', 'same']`:

        - "full": Returns the full convolution result, with shape `(..., M + N - 1)`. (Default)
        - "valid": Returns the segment of the full convolution result corresponding to where
            the two inputs overlap completely, with shape `(..., max(M, N) - min(M, N) + 1)`.
        - "same": Returns the center segment of the full convolution result, with shape `(..., M)`.

    Returns
    -------
    Tensor:
        Convolution result of signal with kernel.
        i.e. with signal of m length and kernel of n length,
        the output length is: For 'full' mode: m+n-1,
        'same' mode: m.
    """
    m = waveform.size(-1)
    n = kernel.size(-1)
    if mode == "full":
        truncate = m + n - 1
    elif mode == "valid":
        truncate = max(m, n) - min(m, n) + 1
    elif mode == "same":
        truncate = m
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Compute convolution using fft.
    padded_size = m + n - 1
    # Round up for cheaper fft.
    fast_ftt_size = next_fast_len(padded_size)
    f_signal = rfft(waveform, n=fast_ftt_size)
    f_kernel = rfft(kernel, n=fast_ftt_size)
    f_result = f_signal * f_kernel
    result = irfft(f_result, n=fast_ftt_size)

    start_idx = (padded_size - truncate) // 2
    return result[..., start_idx : start_idx + truncate]


# Implementation based on speechbrain.
# https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/processing/signal_processing.py
def notch_filter(
    notch_freq: float, filter_width: int = 101, notch_width: float = 0.05
) -> torch.Tensor:
    """
    Returns a notch filter constructed from a high-pass and low-pass filter.

    Arguments
    ---------
    notch_freq:
        frequency to put notch as a fraction of the
        sampling rate / 2. The range of possible inputs is 0 to 1.
    filter_width:
        Filter width in samples. Longer filters have
        smaller transition bands, but are more inefficient.
    notch_width:
        Width of the notch, as a fraction of the sampling_rate / 2.

    Returns
    -------
    notch_filter tensor of shape `[1, 1, filter_width]`
    """

    # Check inputs
    assert 0 < notch_freq <= 1
    assert filter_width % 2 != 0
    pad = filter_width // 2
    inputs = torch.arange(filter_width) - pad

    # Avoid frequencies that are too low
    notch_freq += notch_width

    # Define sinc function, avoiding division by zero
    def sinc(x):
        def _sinc(x):
            return torch.sin(x) / x

        # The zero is at the middle index
        return torch.cat([_sinc(x[:pad]), torch.ones(1), _sinc(x[pad + 1 :])])

    # Compute a low-pass filter with cutoff frequency notch_freq.
    hlpf = sinc(3 * (notch_freq - notch_width) * inputs)
    hlpf *= torch.blackman_window(filter_width)
    hlpf /= torch.sum(hlpf)

    # Compute a high-pass filter with cutoff frequency notch_freq.
    hhpf = sinc(3 * (notch_freq + notch_width) * inputs)
    hhpf *= torch.blackman_window(filter_width)
    hhpf /= -torch.sum(hhpf)
    hhpf[pad] += 1

    # Adding filters creates notch filter
    return (hlpf + hhpf).view(1, 1, -1)


def de_silence(sig: Tensor, sr: int = 16000, win_len: float = 0.1, min_eng: int = 50):
    """
    Trim silence clips in wav according to the average amplitude in windows.

    Arguments
    ---------
    sig:
        The waveform used for deleting silence.
        Shape should be `[channel, time]`.
    sr:
        Sampling rate.
    win_len:
        The window length in "seconds" to compute amplitude.
    min_eng:
        Threshold (not normalized) on the average amplitude on the window.
        If under this threshold, the window is discarded.

    Returns
    -------
    voc_cache: tensor
        The waveform after deleting silence.
    lens:
        Frame number of result waveform.
    """
    duration = len(sig[0])

    voc_block_len = int(win_len * sr)
    min_voc_eng = min_eng / (1 << 15)

    n_win = duration // voc_block_len
    mod = duration % voc_block_len
    voc_cache = []
    lens = 0

    for i in range(n_win):
        if (
            torch.mean(np.abs(sig[0, voc_block_len * i : voc_block_len * (i + 1)]))
            > min_voc_eng
        ):
            voc_cache.append(sig[:, voc_block_len * i : voc_block_len * (i + 1)])
    if mod > 0:
        if torch.mean(np.abs(sig[0, voc_block_len * n_win :])) > min_voc_eng:
            voc_cache.append(sig[:, voc_block_len * n_win :])
    if voc_cache:
        voc_cache = torch.cat(voc_cache, dim=1)
        lens = voc_cache.shape[1]
    return voc_cache, lens


def peak_normalize(waveforms: Tensor, apply_to: str = "only_too_loud_sounds"):
    """
    Samples outside the [-1, 1] range may lead to clipping or wrap distortion,
    This function normalizes a signal to [-1, 1] range.

    Arguments
    ---------
    waveforms:
        The waveforms `[...,time]` to be normalized.
    apply_to:
        Choose between ["all", "only_too_loud_sounds"]. Note: for "all", it applies
        to all wavs, while "only_too_loud_sounds" only applies to wavs that have
        extreme values out side [-1, 1] and leaving others untouched.

    Returns
    -------
    Tensor:
        Normalized level waveform.
    """

    assert apply_to in ["only_too_loud_sounds", "all"]

    abs_max, _ = torch.max(torch.abs(waveforms), dim=-1, keepdim=True)
    if apply_to == "all":
        return waveforms / abs_max.clamp(min=1e-4)
    if apply_to == "only_too_loud_sounds":
        return waveforms / abs_max.clamp(min=1.0)
    else:
        raise NotImplementedError


def calculate_audio_energy(samples, lengths: Optional[Tensor] = None, rms_energy=False):
    """
    Calculates energy of a batch of waveforms, energy could be rms root mean square or mean square.

    Arguments
    ---------
    samples : tensor
        The waveforms used for computing energy.
        Shape should be `[..., time]`.
    lengths : tensor
        The lengths of the waveforms excluding the padding.
        Shape should be a single dimension, `[batch]`.
    rms_energy : bool
        Whether to compute root mean square.

    Returns
    -------
    The energy ( (root) mean square) of the waveforms.

    Example
    -------
    >>> signal = torch.sin(torch.arange(16000.0))
    >>> calculate_audio_energy(signal, rms_energy=True)
    tensor([0.7071])
    """
    if lengths is None:
        out = torch.mean(torch.square(samples), dim=-1, keepdim=True)

    else:
        lengths_shape = lengths.shape + (1,) * (samples.ndim - 1)
        lengths = lengths.view(lengths_shape)
        wav_sum = torch.sum(input=torch.square(samples), dim=-1, keepdim=True)
        out = wav_sum / lengths
    if rms_energy:
        out = torch.sqrt(out)
    return out


def calculate_desired_audio_energy(ref, snr, rms_energy=False):
    """
    Given reference energy of waveforms and signal-to-noise ratio (SNR),
    compute the desired energy.
    Energy could be root mean square (rms) or mean square.

    Arguments
    ---------
    ref : tensor `[batch, ...]`.
        The reference energy.
    snr : tensor `[batch, ...]`.
        desired snr in dB.
    rms_energy : bool
        If True, the input energy level is RMS.

    Returns
    -------
    Tensor: Shape of `[batch, ...]`.
        Target energy level.

    Example
    -------
    >>> ref = torch.tensor([0.4])
    >>> snr=torch.tensor([10])
    >>> calculate_desired_audio_energy(ref,snr,rms_energy=True)
    tensor([0.1265])
    """
    power = 0.5 if rms_energy else 1
    return ref * torch.pow(torch.pow(10.0, 0.1 * snr), -power)


def fast_simulate_rir(
    batch: int = 1,
    nsource: int = 1,
    sample_rate: int = 16000,
    max_D: float = 12.0,
    max_R: float = 1.2,
    direct_rir_compute: bool = False,
    direct_range: Tuple = [-6, 50],
    max_T60: float = 0.8,
    alpha: float = 0.25,
    a: float = -2.0,
    b: float = 2.0,
    tau: float = 0.2,
):
    """
    Simulate room impulse response, the implementation is based on belows but extend it to support batch processing.
        FRA-RIR: Fast Random Approximation of the Image-source Method
        https://arxiv.org/abs/2208.04101
        https://github.com/tencent-ailab/FRA-RIR/blob/main/FRA-RIR.py

    Arguments
    ---------
    batch:
        Batch-wise generater.
    nsource:
        Number of sources (RIR filters) to simulate. Default: 1.
    sample_rate:
        Target sample rate. Default: 16000.
    max_D:
        The maximum range of sample distance between the sound sources and the receiver.
        Default: 12.
    max_R:
        The maximum range of ratio between the volume and the total surface area of the room.
        Default: 1.2, means a room with length, width and height of 12 m, 12 m and 4 m.
                 1.5, length, width and height: 24 m, 24 m and 4 m
                 1.6, length, width and height: 36 m, 36 m and 4 m
                 2.0, length, width and height: 48 m, 36 m and 5 m
    direct_rir_compute:
        If True, output will contain the direct_rir_filter. i.e., early-reverberation-RIR filter. Default: False.
    direct_range:
        The context range (at milliseconds) at the first peak of the RIR filter to define the direct-path RIR. Default: [-6, 50] ms.
    max_T60:
        The maximum range of T60 to sample from. Default: 0.8.
    alpha:
        Controlling the probability distribution to sample the distance of the virtual sound sources from. Default: 0.25.
    a, b:
        Controlling the random pertubation added to each virtual sound source. Default: -2, 2.
    tau:
        Controlling the relationship between the distance and the number of reflections of each virtual sound source. Default: 0.25.

    Returns
    -------
    rir_filter: tensor `[batch, nsource, nsample]`.
        Simulated RIR filter for all sources.
    direct_rir_filter: `[batch, nsource, nsample]`
        Simulated direct-path RIR filter for all sources.

    Example
    -------
    >>> sim_rir, _ = fast_simulate_rir(2)
    >>> max_value, indice = torch.max(torch.abs(sim_rir), dim=-1)
    >>> list(zip(max_value, indice))
    [(tensor([0.0024]), tensor([524])), (tensor([0.0072]), tensor([220]))]
    """

    from .transforms import get_or_create_resampler

    eps = np.finfo(np.float16).eps
    # sample distance between the sound sources and the receiver (d_0)
    direct_dist = torch.FloatTensor(batch, nsource).uniform_(0.2, max_D)
    # sample T60 of the room
    T60 = torch.FloatTensor(batch).uniform_(0.1, max_T60)[..., None]
    # sample room-related statistics for calculating the reflection coefficient R
    R = torch.FloatTensor(batch).uniform_(0.1, max_R)[..., None]
    # number of virtual sound sources
    image = sample_rate * 2
    # the sample rate at which the original RIR filter is generated
    ratio = 64
    sample_sr = sample_rate * ratio
    # sound velocity
    velocity = 340.0
    # indices of direct-path signals based on the sampled d_0
    direct_idx = torch.ceil(direct_dist * sample_sr / velocity).long()
    # length of the RIR filter based on the sampled T60
    rir_length = torch.ceil(sample_sr * T60).long()

    # two resampling operations
    resample1 = get_or_create_resampler(sample_sr, sample_sr // int(np.sqrt(ratio)))
    resample2 = get_or_create_resampler(sample_sr // int(np.sqrt(ratio)), sample_rate)

    # calculate the reflection coefficient based on the Eyring's empirical equation
    reflect_coef = (1 - (1 - torch.exp(-0.16 * R / T60)).pow(2)).sqrt()

    # randomly sample the propagation distance for all the virtual sound sources. (batch, nsources, nsample_image)
    dist_range = (
        torch.linspace(0.0, 1, image) * (velocity * T60 / direct_dist - 1).unsqueeze(-1)
        + 1
    )
    # a simple quadratic function
    dist_prob = torch.linspace(alpha, 1.0, image).pow(2)
    dist_prob = dist_prob / dist_prob.sum()
    dist_select_idx = dist_prob.multinomial(
        num_samples=batch * image * nsource, replacement=True
    ).view(batch, nsource, image)
    # the distance is sampled as a ratio between d_0 and each virtual sound sources
    dist_ratio = torch.gather(dist_range, dim=-1, index=dist_select_idx)
    dist = direct_dist[..., None] * dist_ratio

    # sample the number of reflections (can be nonintegers)
    # calculate the maximum number of reflections
    reflect_max = (
        torch.log10(velocity * T60) - torch.log10(direct_dist) - 3
    ) / torch.log10(reflect_coef + eps)
    # calculate the number of reflections based on the assumption that
    # virtual sound sources which have longer propagation distances may reflect more frequently
    reflect_ratio = (dist / (velocity * T60[..., None])).pow(2) * (
        reflect_max.view(batch, nsource, -1) - 1
    ) + 1
    # add a random pertubation based on the assumption that
    # virtual sound sources which have similar propagation distances can have different routes and reflection patterns
    reflect_pertub = torch.FloatTensor(batch, nsource, image).uniform_(
        a, b
    ) * dist_ratio.pow(tau)
    # all virtual sound sources should reflect for at least once
    reflect_ratio = torch.maximum(reflect_ratio + reflect_pertub, torch.ones(1))

    # calculate the rescaled dirac comb as RIR filter
    dist = torch.cat([direct_dist[..., None], dist], -1)
    reflect_ratio = torch.cat([torch.zeros(batch, nsource, 1), reflect_ratio], -1)

    rir = torch.zeros(batch, nsource, rir_length.max())
    delta_idx = torch.minimum(
        torch.ceil(dist * sample_sr / velocity), rir_length[..., None] - 1
    ).long()
    delta_decay = reflect_coef[..., None].pow(reflect_ratio) / dist
    rir = rir.scatter(dim=-1, index=delta_idx, src=delta_decay)

    # downsample, apply high-pass filter, downsample again
    rir_filter = resample1(rir)
    rir_filter = highpass_biquad(rir_filter, sample_sr // int(np.sqrt(ratio)), 80.0)
    rir_filter = resample2(rir_filter)

    direct_rir_filter = torch.empty(0)
    if direct_rir_compute:
        # create broacasting mask for direct-path RIR
        mask = torch.arange(0, rir_length.max())
        valid_start = torch.maximum(
            direct_idx + sample_sr * direct_range[0] // 1000, torch.tensor(0)
        )
        valid_end = torch.minimum(
            direct_idx + sample_sr * direct_range[1] // 1000, rir_length - 1
        )
        valid_start = valid_start.long()[..., None]
        valid_end = valid_end.long()[..., None]
        valid_lengths = (valid_end - valid_start + 1).flatten().tolist()

        # select valid context range around direct positon
        rir_direct = torch.masked_select(
            rir, (mask >= valid_start) & (mask <= valid_end)
        )
        rir_direct = torch.split(rir_direct, valid_lengths)
        rir_direct = pad_sequence(rir_direct, batch_first=True).view(batch, nsource, -1)

        direct_rir_filter = resample1(rir_direct)
        direct_rir_filter = highpass_biquad(
            direct_rir_filter, sample_sr // int(np.sqrt(ratio)), 80.0
        )
        direct_rir_filter = resample2(direct_rir_filter)
    return rir_filter, direct_rir_filter


class AudioClip:
    """
    Construct signal samples and some info into one class.
    Its `clip` function generate audio clips according to original signal.

    Note: the samples should be a tensor of one wav.
    i.e., shape is `[channels, time]`.
    """

    def __init__(
        self,
        samples: torch.Tensor,
        sample_rate: int,
        id: Optional[str] = None,
        bits_per_sample: Optional[int] = None,
    ):
        self.samples = samples
        self.id = id if id else "temporary"
        self.sample_rate = sample_rate
        self.duration = samples.shape[-1] / self.sample_rate
        self.bits_per_sample = bits_per_sample

    def clip(
        self,
        max_length: Optional[float] = None,
        pad_last_th: Optional[float] = None,
        mean_th: float = 0,  # 5e-4 is the common set of speechbrain.
    ) -> List["AudioClip"]:
        """
        Arguments
        ---------
        max_length:
            if not None, limit the max duration of clips, shorter than it will not be changed.
        pad_last_th:
            if None, the last is a shorter chunk.
            otherwise the parameter means a proportion of max_length,
            the last chunk longger then `max_length * pad_last_th` will be padded with its formmer signals,
            while the shortter will be dropped.
        mean_th:
            The mean L1 norm value of clips lower then this threashould is treated as invalid clip.

        Returns
        -------
        List of `AudioClip`.
        """
        results = []
        if max_length is None or max_length > self.duration:
            results.append(self)
        else:
            # avoid too short.
            assert (
                max_length > 0.2
            ), f"Clip duration {max_length} is shorter than 0.2s, set it longer."
            seg = int(max_length * self.sample_rate)
            num_samples = self.samples.shape[-1]
            cnt = math.ceil(num_samples / seg)
            for i in range(cnt):
                start = seg * i
                end = start + seg
                # managing the last
                if i == cnt - 1:
                    end = num_samples
                    # keep a shorter last
                    if pad_last_th is None:
                        pass
                    # pad the last.
                    elif (num_samples - start) / seg > pad_last_th:
                        start = num_samples - seg
                    # drop last
                    else:
                        continue
                clip_samples = self.samples[:, start:end]

                if mean_th > 0:
                    if clip_samples.is_floating_point():
                        norm = 1
                    else:
                        norm = (
                            (1 << self.bits_per_sample - 1)
                            if self.bits_per_sample
                            else (1 << torch.iinfo(clip_samples.dtype).bits - 1)
                        )
                    if torch.mean(torch.abs(clip_samples[0] / norm)) < mean_th:
                        continue

                clip_id = f"{self.id}_clip{start/self.sample_rate:.2f}_{end/self.sample_rate:.2f}"

                results.append(
                    AudioClip(
                        clip_samples, self.sample_rate, clip_id, self.bits_per_sample
                    )
                )
        return results

    def to_bytes(self):
        f = io.BytesIO()
        torchaudio.backend.soundfile_backend.save(
            f, self.samples, self.sample_rate, format="wav"
        )
        f.seek(0)
        return f

    def __repr__(self):
        return (
            f"AudioClip("
            f"id={self.id}, "
            f"sample_rate={self.sample_rate}, "
            f"duration={round(self.duration, 2)}"
            f")"
        )


def a_law_encoding(x: torch.Tensor, quantization_channels: int = 256) -> torch.Tensor:
    """
    A-law encoding for audio in [-1, 1] → [0, 255] (int64)
    """

    quant = float(quantization_channels - 1)  # 255
    if not x.is_floating_point():
        warnings.warn(
            "The input Tensor must be of floating type. \
            This will be an error in the v0.12 release."
        )
        x = x.to(torch.float)

    device = x.device
    dtype = x.dtype

    abs_x = x.abs()
    sign_x = x.sign()

    A = 87.6
    A_tensor = torch.tensor(A, device=device, dtype=dtype)
    x_narrow = A_tensor * abs_x
    x_wide = 1 + torch.log(x_narrow)
    x_numerator = torch.where(abs_x < (1 / A_tensor), x_narrow, x_wide)

    x_a = sign_x * x_numerator / (1.0 + torch.log(A_tensor))

    # 映射到 [0, 255] 并四舍五入
    x_a = ((x_a + 1.0) / 2.0 * quant + 0.5).to(torch.int64)
    return x_a


def a_law_decoding(x_q: torch.Tensor, quantization_channels: int = 256) -> torch.Tensor:
    """
    A-law decoding: [0, 255] → [-1, 1]
    """
    quant = float(quantization_channels - 1)  # 255
    if not x_q.is_floating_point():
        x_q = x_q.to(torch.float)
    A = 87.6
    A_tensor = torch.tensor(A, device=x_q.device, dtype=x_q.dtype)
    x_a = (x_q / quant) * 2 - 1.0  # [0,255] -> [-1, 1]
    ln_a = 1 + torch.log(A_tensor)
    x_abs = torch.abs(x_a)
    x_narrow = x_abs * ln_a
    x_wide = torch.exp(x_narrow - 1)
    x_numerator = torch.where(x_abs < (1 / ln_a), x_narrow, x_wide)
    x = torch.sign(x_a) * x_numerator / A_tensor
    return x
