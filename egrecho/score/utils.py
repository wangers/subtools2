# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-09)

import warnings
from itertools import chain
from typing import Iterable

import numpy as np


def compute_mean_stats(data: Iterable[np.ndarray]):
    data_iter = iter(data)
    first = next(data_iter)
    total_utts = 0
    data_type = first.dtype
    total_sum = np.zeros_like(first, dtype=np.float64)
    for vec in chain([first], data_iter):
        vec = vec.astype(np.float64)
        total_sum += vec
        total_utts += 1
    mean = total_sum / total_utts
    return mean.astype(data_type)


def cumsum(arr, axis=None, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum.

    Warns if the final cumulative sum does not match the sum (up to the chosen
    tolerance).

    Args:
        arr : array-like
            To be cumulatively summed as flat.
        axis : int, default=None
            Axis along which the cumulative sum is computed.
            The default (None) is to compute the cumsum over the flattened array.
        rtol : float, default=1e-05
            Relative tolerance, see ``np.allclose``.
        atol : float, default=1e-08
            Absolute tolerance, see ``np.allclose``.

    Returns
        out : ndarray
            Array with the cumulative sums along the chosen axis.
    """
    out = np.cumsum(arr, axis=axis, dtype=np.float64)
    expected = np.sum(arr, axis=axis, dtype=np.float64)
    if not np.allclose(
        out.take(-1, axis=axis), expected, rtol=rtol, atol=atol, equal_nan=True
    ):
        warnings.warn(
            (
                "cumsum was found to be unstable: "
                "its last element does not correspond to sum"
            ),
            RuntimeWarning,
        )
    return out
