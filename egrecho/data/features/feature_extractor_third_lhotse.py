# -*- coding:utf-8 -*-
# (Author: Leo 2024-09)

"""Feats from third party lhotse."""

import copy
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from egrecho.data.features.feature_extractor_audio import (
    LOG_EPSILON,
    BatchTensor,
    SequenceFeature,
    SingleTensor,
    cmvn_utts,
)
from egrecho.utils.common import alt_none, dict_union
from egrecho.utils.imports import _LHOTSE_AVAILABLE
from egrecho.utils.logging import get_logger
from egrecho.utils.misc import ConfigurationException, add_end_docstrings
from egrecho.utils.torch_utils import to_torch_tensor

if not _LHOTSE_AVAILABLE:
    raise ImportError(
        "To use ExtLhotseFeatureExtractor,  please ``pip install lhotse`` first."
    )
from lhotse.features.base import FEATURE_EXTRACTORS, FeatureExtractor

logger = get_logger(__name__)
DEFAULT_FEAT_CONF = {"feature_type": "kaldi-fbank"}


EXTLHOTSE_EXAMPLE_DOCSTRING = r"""
        Example:

        .. code-block::

            import torch
            from egrecho.data.features.feature_extractor_third_lhotse import ExtLhotseFeatureExtractor

            extractor = ExtLhotseFeatureExtractor(mean_norm=False, feat_conf={})
            signal = torch.sin(torch.arange(16000))
            outs = extractor(signal, sampling_rate=16000)
            feats, att_mask = outs['input_features'], outs['attention_mask']
            assert tuple(feats.shape) == (1, 100, 80)
            assert (att_mask.sum(dim=-1).tolist() == [100])

        .. code-block::

            outs
            {'input_features': tensor([[[ -7.1169, ..., -2.6014],
                    [ -9.5259, ..., -15.9424],
                    ...,
                    [ -6.9482, ..., -2.7877]]]),
            'attention_mask': tensor([[1, ..., 1]], dtype=torch.int32)}

        .. code-block::

            fbank = (torch.randn(100, 80), torch.randn(75, 80))

            outs = extractor(fbank, sampling_rate=16000, offline_feats=True)
            feats, att_mask = outs['input_features'], outs['attention_mask']
            assert tuple(feats.shape) == (2, 100, 80)
            assert (att_mask.sum(dim=-1).tolist() == [100,75])
"""
EXTLHOTSE_EXAMPLE_DOCSTRING_CLS = EXTLHOTSE_EXAMPLE_DOCSTRING.replace(
    "\n        ",
    "\n    ",
)


@lru_cache(maxsize=8)
def get_ext_lhotse_feat(**feat_conf) -> FeatureExtractor:
    return FeatureExtractor.from_dict(feat_conf)


@add_end_docstrings(EXTLHOTSE_EXAMPLE_DOCSTRING_CLS)
class ExtLhotseFeatureExtractor(SequenceFeature):
    """
    Third package lhotse extractor class, inheriting from SequenceFeature.

    Args:
        sampling_rate (Optional[int]):
            The input sampling rate. Defaults to 16000
        feat_conf (Optional[Dict]):
            Feature configuration. Defaults to kaldi-fbank
        scale_bit Optional[int]:
            Whether scale inputs signal to 16/32 bit.
        mean_norm (bool):
            Whether to perform mean normalization on feature. Defaults to False
        std_norm (bool):
            Whether to perform standard deviation normalization. Defaults to False
        return_attention_mask (bool):
            Whether to return attention mask.
        padding_value (float):
            Padding value, default is log of `1e-10`.
        rich_feat_info (bool):
            Whether to include rich feature information. Defaults to False.
        **kwargs:
            Other arguments.
    """

    # Input names expected by the model
    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        sampling_rate: Optional[int] = None,
        feat_conf: Optional[Dict] = None,
        scale_bit: Optional[int] = None,
        mean_norm: bool = False,
        std_norm: bool = False,
        return_attention_mask: bool = True,
        padding_value: float = LOG_EPSILON,
        rich_feat_info: bool = False,
        **kwargs,
    ):

        if kwargs.get("feature_size"):
            raise ConfigurationException(
                "`feature_size` is just for compatible reason, it is not allowed for passing "
                f"as an argument, but got ({kwargs.get('feature_size')}), set `feat_conf` to control feature dim."
            )

        # sequence defaults sampling_rate
        sr = {"sampling_rate": sampling_rate} if sampling_rate is not None else {}
        self.feat_conf = dict_union(DEFAULT_FEAT_CONF, sr, alt_none(feat_conf, {}))
        self.scale_bit = scale_bit

        if rich_feat_info:
            self.feat_conf = get_ext_lhotse_feat(**feat_conf).to_dict()

        extactor = self.extractor
        sampling_rate = extactor.config.sampling_rate
        feature_size = extactor.feature_dim(sampling_rate)

        self.mean_norm = mean_norm
        self.std_norm = std_norm

        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )

    @add_end_docstrings(EXTLHOTSE_EXAMPLE_DOCSTRING)
    def __call__(
        self,
        input_values: Union[SingleTensor, BatchTensor],
        sampling_rate: Optional[int] = None,
        max_length: Optional[int] = None,
        truncate: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: bool = True,
        offline_feats: bool = False,
    ) -> dict:
        """
        Call the feature extractor for feature extraction.

        Args:
            input_values (Union[SingleTensor, BatchTensor]): Input input_values, can be either single tensor or
                batch of tensors.

                    - audio: ``offline_feats=False``, shape of ``[C, T]``|``[T,]`` for each
                    - offline feats: ``offline_feats=True``, shape of ``[T, F]`` for each

            sampling_rate (Optional[int]):
                Input sample rate. passing in this function aims to do checking
                for the same with configuration of extractor.
            max_length (`int`, *optional*):
                fix the maximum length of the returned list if `truncate=True`.
            truncate (`bool`, *optional*):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            return_attention_mask (Optional[bool]):
                Whether to return attention mask. if specified, will affect the default set in `__init__`.
            return_tensors (bool):
                If True, output features in batch is a tensor, otherwise list of tensors. Defaults to True.
            offline_feats (bool):
                Whether accept offline feature inputs or not.

        Returns:
            dict: A dictionary containing features.
        """
        if not offline_feats:
            return self._call_raw(
                samples=input_values,
                sampling_rate=sampling_rate,
                max_length=max_length,
                truncate=truncate,
                return_attention_mask=return_attention_mask,
                return_tensors=return_tensors,
            )
        else:
            return self._call_offline(
                features=input_values,
                sampling_rate=sampling_rate,
                max_length=max_length,
                truncate=truncate,
                return_attention_mask=return_attention_mask,
                return_tensors=return_tensors,
            )

    def _call_raw(
        self,
        samples: Union[SingleTensor, BatchTensor],
        sampling_rate: Optional[int] = None,
        max_length: Optional[int] = None,
        truncate: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: bool = True,
    ) -> dict:
        """
        Call the feature extractor for feature extraction. `(C, T)/(T,)` -> `(B, T, F)`

        Args:
            input_values (Union[SingleTensor, BatchTensor]):
                Input input_values, can be either single tensor audio of shape `[C, T]`/`[C, T]` or batch of tensors.
            sampling_rate (Optional[int]):
                Input sample rate. passing in this function aims to do checking
                for the same with configuration of extractor.
            max_length (`int`, *optional*):
                fix the maximum length of the returned list if `truncate=True`.
            truncate (`bool`, *optional*):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            return_attention_mask (Optional[bool]):
                Whether to return attention mask. if specified, will affect the default set in `__init__`.
            return_tensors (bool):
                If True, output features in batch is a tensor, otherwise list of tensors. Defaults to True.

        Returns:
            dict: A dictionary containing features.

        Example:
        >>> extractor = ExtLhotseFeatureExtractor(mean_norm=False)
        >>> signal = torch.sin(torch.arange(16000))
        >>> extractor._call_raw(signal,sampling_rate=16000)
        {'input_features': tensor([[[ -7.1169, ..., -2.6014],
                [ -9.5259, ..., -15.9424],
                ...,
                [ -6.9482, ..., -2.7877]]]),
        'attention_mask': tensor([[1, ..., 1]], dtype=torch.int32)}
        """
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self.__class__.__name__} was trained using a"
                    f" sampling rate of {self.sampling_rate}. Please make sure that the provided `samples` input"
                    f" was sampled with {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                "It is strongly recommended to pass the `sampling_rate` argument to this function. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )
            sampling_rate = self.sampling_rate

        is_batched = bool(
            isinstance(samples, (list, tuple))
            and (
                isinstance(samples[0], np.ndarray)
                or isinstance(samples[0], torch.Tensor)
            )
        )
        # always return batch
        if not is_batched:
            samples = [samples]
        if self.scale_bit:
            mov = int(self.scale_bit - 1)
            samples = [sample * (1 << mov) for sample in samples]
        extractor: FeatureExtractor = get_ext_lhotse_feat(**self.feat_conf)
        feats = extractor.extract_batch(samples, sampling_rate=sampling_rate)
        if not isinstance(feats, list):
            feats = [feat for feat in feats]

        do_norm = self.mean_norm or self.std_norm
        batched_feats = {"input_features": feats}
        return_mask = do_norm if do_norm else return_attention_mask
        padded_inputs = self.pad(
            batched_feats,
            max_length=max_length,
            truncate=truncate,
            return_attention_mask=return_mask,
        )
        if do_norm:
            padded_inputs["input_features"] = cmvn_utts(
                padded_inputs["input_features"],
                attention_mask=padded_inputs["attention_mask"],
                padding_value=self.padding_value,
            )
            if not bool(return_attention_mask) and not self.return_attention_mask:
                padded_inputs.pop("attention_mask", None)

        if not return_tensors:
            padded_inputs["input_features"] = [feat for feat in padded_inputs]
        return padded_inputs

    def _call_offline(
        self,
        features: Union[SingleTensor, BatchTensor],
        sampling_rate: Optional[int] = None,
        max_length: Optional[int] = None,
        truncate: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: bool = True,
    ) -> dict:
        """
        Offline feature extractor. `(T, F) -> (B, T, F)`

        Args:
            features (Union[SingleTensor, BatchTensor]):
                Input features, can be either single tensor feature of shape `[T, C]` or batch of tensors.
            return_attention_mask (Optional[bool]):
                Whether to return attention mask. if specified, will affect the default set in `__init__`.
            max_length (`int`, *optional*):
                fix the maximum length of the returned list if `truncate=True`.
            truncate (`bool`, *optional*):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            return_tensors (bool):
                If True, output features in batch is a tensor, otherwise list of tensors. Defaults to True.

        Returns:
            dict: A dictionary containing features.

        Example::

            import torch

            from egrecho.data.features.feature_extractor_third_lhotse import ExtLhotseFeatureExtractor

            fbank = (torch.randn(100, 80), torch.randn(75, 80))
            extractor = ExtLhotseFeatureExtractor(mean_norm=False, feat_conf={})
            outs = extractor._call_offline(fbank, sampling_rate=16000)
            feats, att_mask = outs['input_features'], outs['attention_mask']
            assert tuple(feats.shape) == (2, 100, 80)
            assert (att_mask.sum(dim=-1).tolist() == [100,75])
        """
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self.__class__.__name__} was trained using a"
                    f" sampling rate of {self.sampling_rate}. Please make sure that the provided `samples` input"
                    f" was sampled with {self.sampling_rate} and not {sampling_rate}."
                )

        is_batched = bool(
            isinstance(features, (list, tuple))
            and (
                isinstance(features[0], np.ndarray)
                or isinstance(features[0], torch.Tensor)
            )
        )
        # always return batch
        if not is_batched:
            features = [features]
        features = [to_torch_tensor(feat) for feat in features]
        if features[0].shape[1] != self.feature_size:
            raise ConfigurationException(
                "The expected feature size, as specified in your feats-extractor (feat_conf), "
                f"should match for lenient checking, but got feature size of ({features[0].shape[1]}) "
                f"does not match self.feature_size ({self.feature_size})."
            )

        do_norm = self.mean_norm or self.std_norm
        batched_feats = {"input_features": features}
        return_mask = do_norm if do_norm else return_attention_mask
        padded_inputs = self.pad(
            batched_feats,
            max_length=max_length,
            truncate=truncate,
            return_attention_mask=return_mask,
        )
        if do_norm:
            padded_inputs["input_features"] = cmvn_utts(
                padded_inputs["input_features"],
                attention_mask=padded_inputs["attention_mask"],
                padding_value=self.padding_value,
            )
            if not bool(return_attention_mask) and not self.return_attention_mask:
                padded_inputs.pop("attention_mask", None)

        if not return_tensors:
            padded_inputs["input_features"] = [feat for feat in padded_inputs]
        return padded_inputs

    @property
    def extractor(self):
        return get_ext_lhotse_feat(**self.feat_conf)

    @property
    def feature_dim(self) -> int:
        return self.feature_size

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`.
        """

        output = copy.deepcopy(self.__dict__)
        output["feature_extractor_type"] = self.__class__.__name__
        output.pop("feature_size", None)
        output.pop("sampling_rate", None)
        return output

    @classmethod
    def available_extractors(self) -> List[str]:
        return FEATURE_EXTRACTORS.keys()
