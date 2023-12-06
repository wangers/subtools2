# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-09)

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from egrecho.score.utils import cumsum


def compute_metrics(
    y_score: np.ndarray,
    y_true: np.ndarray,
    p_target: float,
    c_miss: float = 1,
    c_fa: float = 1,
) -> Tuple[float, float, float]:
    """Computes EER & minDCF.

    Args:
        y_score: 1d array with predic scores.
        y_true : 1d array with true values
        p_target: Prior probability for target speakers.
        c_miss: Cost associated with a missed detection (default is 1).
        c_fa: Cost associated with a false alarm (default is 1).

    Returns:
        A tuple contains EER, minDCT, EER threshold.
    """
    fprs, fnrs, thresholds = det_curve(y_score, y_true)
    eer, eer_threshold = eer_processor(fprs, fnrs, thresholds)
    min_dcf = min_dcf_processor(fprs, fnrs, p_target=p_target, c_miss=c_miss, c_fa=c_fa)

    return eer, min_dcf, eer_threshold


def det_curve(
    y_score: np.ndarray,
    y_true: np.ndarray,
    sample_weights: Optional[Union[Sequence, np.ndarray]] = None,
    pos_label: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute error rates for different probability thresholds.

    Args:
        y_score: 1d array with predic scores.
        y_true : 1d array with true values
        sample_weights: a 1d array with a weight per sample
        pos_label: integer determining what the positive class in target tensor is

    Returns:
        fpr: 1d array with false positives rate (decreasing) for different thresholds
        fnr: 1d array with false negatives rate (increasing) for different thresholds
        thresholds: score values (increasing) as the unique thresholds
    """
    fps, tps, thresholds = _binary_clf_curve(
        y_score, y_true, pos_label=pos_label, sample_weights=sample_weights
    )
    fns = tps[-1] - tps

    p_count = tps[-1]
    n_count = fps[-1]
    # start with false positives zero
    first_idx = (
        fps.searchsorted(fps[0], side="right") - 1
        if fps.searchsorted(fps[0], side="right") > 0
        else None
    )
    # stop with false negatives zero
    last_idx = tps.searchsorted(tps[-1]) + 1
    sl = slice(first_idx, last_idx)

    # reverse the output such that list of false positives is decreasing
    return (fps[sl][::-1] / n_count, fns[sl][::-1] / p_count, thresholds[sl][::-1])


def eer_processor(
    fprs: np.ndarray,
    fnrs: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[float, float]:
    """Computes EER (equal error rates) from fprs, fnrs, threasholds.

    Args:
        fprs: 1d array with false positives rate (decreasing) for different thresholds
        fnrs: 1d array with false negatives rate (increasing) for different thresholds
        thresholds: score values (increasing) as the unique thresholds

    Returns:
        Tuple containing EER and its corresponding threshold score.
    """

    # Check if arrays are not empty
    if len(fprs) == 0 or len(fnrs) == 0:
        raise ValueError("Input arrays must not be empty.")

    # Find the indices where the curves intersect
    intersec_start_idx = np.flatnonzero(fnrs - fprs <= 0)[-1]
    intersec_end_idx = np.flatnonzero(fnrs - fprs > 0)[0]

    # Calculate EER and its corresponding threshold
    start_diff = fnrs[intersec_start_idx] - fprs[intersec_start_idx]  # negative
    end_diff = fnrs[intersec_end_idx] - fprs[intersec_end_idx]  # positive
    scale = abs(start_diff) / (end_diff - start_diff)

    # return eer, threshold
    eer = fnrs[intersec_start_idx] + scale * (
        fnrs[intersec_end_idx] - fnrs[intersec_start_idx]
    )
    threshold = thresholds[intersec_start_idx] + scale * (
        thresholds[intersec_end_idx] - thresholds[intersec_start_idx]
    )

    return eer, threshold


def min_dcf_processor(
    fprs: np.ndarray,
    fnrs: np.ndarray,
    p_target: float,
    c_miss: float = 1,
    c_fa: float = 1,
) -> float:
    """
    Computes the minDCF (Normalized Minimum Detection Cost Function).

    Args:
        fprs: 1D array of false positive rates (decreasing) for different thresholds.
        fnrs: 1D array of false negative rates (increasing) for different thresholds.
        p_target: Prior probability for target speakers.
        c_miss: Cost associated with a missed detection (default is 1).
        c_fa: Cost associated with a false alarm (default is 1).

    Returns:
        Normalized Minimum Detection Cost.
    """

    # Calculate the detection cost and the cost of false alarm
    detection_cost = c_miss * fnrs * p_target + c_fa * fprs * (1 - p_target)
    false_cost = min(c_miss * p_target, c_fa * (1 - p_target))

    # Find the minimum detection cost
    min_detection_cost = np.min(detection_cost)

    # Normalize the minimum detection cost
    normalized_detection_cost = min_detection_cost / false_cost

    return normalized_detection_cost


def _binary_clf_curve(
    y_score: np.ndarray,
    y_true: np.ndarray,
    sample_weights: Optional[Union[Sequence, np.ndarray]] = None,
    pos_label: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the true and false positives for all unique thresholds.

    Referring:
        https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_ranking.py.

    Args:
        y_score: 1d array with predic scores.
        y_true : 1d array with true values
        sample_weights: a 1d array with a weight per sample
        pos_label: integer determining what the positive class in y_true is

    Returns:
        fps: 1d array with false positives (increasing) for different thresholds
        tps: 1d array with true positives (increasing) for different thresholds
        thresholds: score values (decreasing) as the unique thresholds
    """
    if sample_weights is not None and not isinstance(sample_weights, np.ndarray):
        sample_weights = np.array(sample_weights, dtype=np.float64)

    if not (y_score.ndim == y_true.ndim):
        raise ValueError(
            f"y should be a 1d array, got array of shape y_true={y_true.shape}, y_true={y_score.shape}, "
        )
    if sample_weights is not None:
        assert sample_weights.ndim == 1
        nonzero_weight_mask = sample_weights != 0
        y_true = y_true[nonzero_weight_mask]
        y_score = y_score[nonzero_weight_mask]
        sample_weights = sample_weights[nonzero_weight_mask]

    y_true = y_true == pos_label
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]

    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    weight = sample_weights[desc_score_indices] if sample_weights is not None else 1.0

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.

    # find positions where value diffs.
    distinct_value_indices = np.where(y_score[1:] - y_score[:-1])[0]
    # concat a largest idx to the end.
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = cumsum(y_true * weight)[threshold_idxs]

    if sample_weights is not None:
        fps = cumsum((1 - y_true) * weight)[threshold_idxs]
    else:
        fps = 1 + threshold_idxs - tps

    return fps, tps, y_score[threshold_idxs]
