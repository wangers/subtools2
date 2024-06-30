# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2024-03)

from typing import Optional

import torch


def subsequent_chunk_mask(
    size: int,
    chunk_size: int,
    num_left_chunks: int = -1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size) with chunk size

    Args:
        size (int): size of mask
        chunk_size (int): size of chunk
        num_left_chunks (int): number of left chunks
            <0: use full chunk
            >=0: use num_left_chunks
        device (torch.device): "cpu" or "cuda" or torch.Tensor.device

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_chunk_mask(4, 2)
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    """
    ret = torch.zeros(size, size, device=device, dtype=torch.bool)
    for i in range(size):
        if num_left_chunks < 0:
            start = 0
        else:
            start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)
        ending = min((i // chunk_size + 1) * chunk_size, size)
        ret[i, start:ending] = True
    return ret


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.BoolTensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.BoolTensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    if isinstance(lengths, (list, tuple)):
        lengths = torch.tensor(lengths)
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def make_non_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.BoolTensor:
    """Make mask tensor containing indices of non-padded part.

    The sequences in a batch may have different lengths. To enable
    batch computing, padding is need to make all sequence in same
    size. To avoid the padding part pass value to context dependent
    block such as attention or convolution , this padding part is
    masked.

    This pad_mask is used in both encoder and decoder.

    1 for non-padded part and 0 for padded part.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.BoolTensor: mask tensor containing indices of no padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1 ,1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
    """
    return ~make_pad_mask(lengths, max_len=max_len)


def prepare_4d_attention_mask(
    mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None
):
    """
    Creates a non-causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`. The values in org 2D mask is 0 (no-padded) or 1 (padded).

    Args:
        mask (`torch.Tensor` or `None`):
            A 2D attention mask of shape `(batch_size, key_value_length)`
        dtype (`torch.dtype`):
            The torch dtype the created mask shall have.
        tgt_len (`int`):
            The target length or query length the created mask shall have.
    """
    return _expand_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


def make_causal_mask(
    inputs: torch.Tensor,
    dtype: torch.dtype,
    past_key_values_length: int = 0,
    sliding_window: Optional[int] = None,
):
    """
    Creates a causal 4d mask from given querys.

    Args:
        inputs: inputs query of shape `[bsz, seq_len, ...]`.
        dtype: the torch dtype the created mask shall have.
        past_key_values_length: cached past kv length
        sliding_window: left chunk size

    Returns:
        4d attention mask (batch_size, 1, query_length, key_value_length) that can be multiplied with attention scores.

    Examples:
        ```python
        >>> make_causal_mask(torch.arange(3).view(1,3),torch.float)
        tensor([[[[ 0.0000e+00, -3.4028e+38, -3.4028e+38],
                [ 0.0000e+00,  0.0000e+00, -3.4028e+38],
                [ 0.0000e+00,  0.0000e+00,  0.0000e+00]]]])
        >>> make_causal_mask(torch.arange(3).view(1,3), torch.float, past_key_values_length=2)
        tensor([[[[ 0.0000e+00,  0.0000e+00,  0.0000e+00, -3.4028e+38, -3.4028e+38],
                [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -3.4028e+38],
                [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00]]]])
        >>> make_causal_mask(torch.arange(3).view(1,3), torch.float, past_key_values_length=2, sliding_window=2)
        tensor([[[[ 0.0000e+00,  0.0000e+00,  0.0000e+00, -3.4028e+38, -3.4028e+38],
                [-3.4028e+38,  0.0000e+00,  0.0000e+00,  0.0000e+00, -3.4028e+38],
                [-3.4028e+38, -3.4028e+38,  0.0000e+00,  0.0000e+00,  0.0000e+00]]]])
        ```
    """

    input_ids_shape = inputs.shape[:2]  # bsz, seq_len

    return _make_causal_mask(
        input_ids_shape, dtype, inputs.device, past_key_values_length, sliding_window
    )


def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
    sliding_window: Optional[int] = None,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )

    # add lower triangular sliding window mask if necessary
    if sliding_window is not None:
        diagonal = past_key_values_length - sliding_window - 1

        context_mask = torch.tril(
            torch.ones_like(mask, dtype=torch.bool), diagonal=diagonal
        )
        mask.masked_fill_(context_mask, torch.finfo(dtype).min)

    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )
