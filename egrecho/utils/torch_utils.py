# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-02)

from collections import UserDict
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

import numpy as np
import torch


@dataclass
class RandomValue:
    """
    Generate a uniform distribution in the range `[start, end]`.
    """

    end: Union[int, float]
    start: Union[int, float] = 0

    def __post_init__(self):
        assert self.start <= self.end

    def sample(
        self, shape: Union[Sequence[int], int] = [], device: str = "cpu"
    ) -> torch.Tensor:
        rad = torch.rand(shape, device=device)
        return (self.end - self.start) * rad + self.start

    def sample_int(
        self, shape: Union[Sequence[int], int] = [], device: str = "cpu"
    ) -> torch.Tensor:
        assert self.start <= self.end
        start = int(self.start)
        end = int(self.end)
        if isinstance(shape, int):
            shape = tuple([shape])
        return torch.randint(low=start, high=end + 1, size=shape, device=device)


# https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/utils/data_utils.py
def batch_pad_right(tensors: list, mode="constant", value=0, val_index=-1):
    """Given a list of torch tensors it batches them together by padding to the right
    on each dimension in order to get same length for all.

    Parameters
    ----------
    tensors : list
        List of tensor we wish to pad together.
    mode : str
        Padding mode see torch.nn.functional.pad documentation.
    value : float
        Padding value see torch.nn.functional.pad documentation.

    Returns
    -------
    tensor : torch.Tensor
        Padded tensor.
    valid_vals : list
        List containing proportion for each dimension of original, non-padded values.

    """

    if not len(tensors):
        raise IndexError("Tensors list must not be empty")
    # tensors = list(map(list,tensors))

    if len(tensors) == 1:
        # if there is only one tensor in the batch we simply unsqueeze it.
        return tensors[0].unsqueeze(0), torch.tensor([1.0])

    if not (any([tensors[i].ndim == tensors[0].ndim for i in range(1, len(tensors))])):
        raise IndexError("All tensors must have same number of dimensions")

    # FIXME we limit the support here: we allow padding of only the last dimension
    # need to remove this when feat extraction is updated to handle multichannel.
    max_shape = []
    for dim in range(tensors[0].ndim):
        if dim != (tensors[0].ndim - 1):
            if not all([x.shape[dim] == tensors[0].shape[dim] for x in tensors[1:]]):
                raise EnvironmentError(
                    "Tensors should have same dimensions except for last one"
                )
        max_shape.append(max([x.shape[dim] for x in tensors]))

    batched = []
    valid = []
    for t in tensors:
        # for each tensor we apply pad_right_to
        padded, valid_percent = pad_right_to(t, max_shape, mode=mode, value=value)
        batched.append(padded)
        valid.append(valid_percent[val_index])

    batched = torch.stack(batched)

    return batched, torch.tensor(valid)


def pad_right_to(
    tensor: torch.Tensor,
    target_shape: (list, tuple),
    mode="constant",
    value=0,
):
    """
    This function takes a torch tensor of arbitrary shape and pads it to target
    shape by appending values on the right.

    Parameters
    ----------
    tensor : input torch tensor
        Input tensor whose dimension we need to pad.
    target_shape : (list, tuple)
        Target shape we want for the target tensor its len must be equal to tensor.ndim
    mode : str
        Pad mode, please refer to torch.nn.functional.pad documentation.
    value : float
        Pad value, please refer to torch.nn.functional.pad documentation.

    Returns
    -------
    tensor : torch.Tensor
        Padded tensor.
    valid_vals : list
        List containing proportion for each dimension of original, non-padded values.
    """
    assert len(target_shape) == tensor.ndim
    # this contains the abs length of the padding for each dimension.
    pads = []
    valid_vals = []  # thic contains the relative lengths for each dimension.
    i = len(target_shape) - 1  # iterating over target_shape ndims
    j = 0
    while i >= 0:
        assert (
            target_shape[i] >= tensor.shape[i]
        ), "Target shape must be >= original shape for every dim"
        pads.extend([0, target_shape[i] - tensor.shape[i]])
        valid_vals.append(tensor.shape[j] / target_shape[j])
        i -= 1
        j += 1
    tensor = torch.nn.functional.pad(tensor, pads, mode=mode, value=value)

    return tensor, valid_vals


def audio_collate_fn(
    waveforms: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of waves with shape `(..., T)`, returns a tuple containing tensor and a list of its lengths."""
    lengths = torch.tensor(
        [waveform.size(-1) for waveform in waveforms], dtype=torch.int32
    )

    # List[(C, T)] -> List[(T, C)] -> Tensor[B, T, C]
    waveforms = torch.nn.utils.rnn.pad_sequence(
        [waveform.transpose(-2, -1) for waveform in waveforms], batch_first=True
    )
    return waveforms.transpose(-2, -1), lengths


def to_numpy(obj):
    """
    Convert a  PyTorch tensor, Numpy array or python list to a Numpy array.
    """
    if isinstance(obj, (dict, UserDict)):
        return {k: to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return np.array(obj)
    elif torch.is_tensor(obj):
        return obj.detach().cpu().numpy()
    else:
        return obj


def to_torch_tensor(obj):
    if isinstance(obj, (dict, UserDict)):
        return {k: to_torch_tensor(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return torch.tensor(obj)
    elif isinstance(obj, np.ndarray):
        return torch.as_tensor(obj)
    else:
        return obj


def tensor_has_nan(tensor: torch.Tensor):
    if torch.any((torch.isnan(tensor))):
        return True
    return False
