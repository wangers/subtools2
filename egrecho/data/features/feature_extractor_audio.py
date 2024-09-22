# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-07)

import collections
import copy
import math
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate

from egrecho.core.feature_extractor import BaseFeature
from egrecho.data.features.lhotse_kaldi import FEATURE_EXTRACTORS, LhotseFeat
from egrecho.utils.common import alt_none, dict_union
from egrecho.utils.logging import get_logger
from egrecho.utils.misc import ConfigurationException
from egrecho.utils.torch_utils import to_torch_tensor

logger = get_logger(__name__)

SingleTensor = Union[torch.Tensor, np.ndarray]
BatchTensor = Union[Sequence[np.ndarray], Sequence[torch.Tensor]]

EPSILON = 1e-10
LOG_EPSILON = math.log(EPSILON)
DEFAULT_FEAT_CONF = {"feature_type": "kaldi-fbank"}


@lru_cache(maxsize=8)
def get_lhotse_feat(**feat_conf) -> LhotseFeat:
    return LhotseFeat.from_dict(feat_conf)


class SequenceFeature(BaseFeature):
    """
    This is a general feature extraction class for speech inputs.

    The main objective of this base class is to provide padding and collate methods for speech inputs.

    Args:
        feature_size (int):
            The feature dimension of the extracted features.
        sampling_rate (int):
            The sampling rate at which the audio files should be digitized, expressed in hertz (Hz).
        padding_value (float):
            The value used to fill the padding values/vectors.
    """

    def __init__(
        self, feature_size: int, sampling_rate: int, padding_value: float, **kwargs
    ):
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value

        self.padding_side = kwargs.pop("padding_side", "right")
        self.return_attention_mask = kwargs.pop("return_attention_mask", True)

        super().__init__(**kwargs)

    def pad(
        self,
        processed_features: Union[
            Mapping[str, List],
            List[Mapping[str, Any]],
        ],
        max_length: Optional[int] = None,
        truncate: bool = False,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad a batch of input values / input vectors up to the max sequence length in the batch.

        Padding side (left/right) padding values are defined at the feature extractor level (with `self.padding_side`,
        `self.padding_value`)

        Args:
            processed_features:
                ([`Mapping`] with list of batch, list of examples of [`Mapping`].
                you can use this method during preprocessing as well as in a PyTorch Dataloader collate function.
            max_length (`int`, *optional*):
                fix the maximum length of the returned list if `truncate=True`.
            truncate (`bool`, *optional*):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.
        """
        # If we have a list of dicts, let's convert it in a dict of lists
        # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
        if isinstance(processed_features, (list, tuple)) and isinstance(
            processed_features[0], collections.abc.Mapping
        ):
            processed_features = {
                key: [example[key] for example in processed_features]
                for key in processed_features[0].keys()
            }

        # check the model's main input name
        if self.model_input_names[0] not in processed_features:
            raise ValueError(
                f"Batch passed to this function should be a list of dict or a dict contains list of dict"
                f" include {self.model_input_names[0]} but you provide"
                f" {list(processed_features.keys())}"
            )

        required_input = processed_features[self.model_input_names[0]]
        return_attention_mask = (
            return_attention_mask
            if return_attention_mask is not None
            else self.return_attention_mask
        )

        if not required_input:
            if return_attention_mask:
                processed_features["attention_mask"] = []
            return processed_features

        batch_size = len(required_input)
        if not all(len(v) == batch_size for v in processed_features.values()):
            raise ValueError(
                "Some items in the output dictionary have a different batch size than others."
            )

        # convert tensor to pad
        processed_features[self.model_input_names[0]] = [
            to_torch_tensor(x) for x in required_input
        ]

        # break down into list of dicts on the main column for padding.
        inputs_to_pad = []
        for i in range(batch_size):
            input_to_cut = {k: v[i] for k, v in processed_features.items()}
            inputs_slice = self._truncate(
                input_to_cut,
                max_length=max_length,
                truncate=truncate,
            )
            inputs_to_pad.append(inputs_slice)
        max_length = max(
            (example[self.model_input_names[0]]).shape[0] for example in inputs_to_pad
        )

        batch_outputs = {}
        for i in range(batch_size):
            # padding
            outputs = self._pad(
                inputs_to_pad[i],
                max_length=max_length,
                return_attention_mask=return_attention_mask,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        for key, value in batch_outputs.items():
            batch_outputs[key] = default_collate(value)
        return batch_outputs

    def _pad(
        self,
        processed_features: Dict[str, torch.Tensor],
        max_length: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad single inputs (on left/right in the batch)

        Args:
            processed_features (`Dict[str, torch.Tensor]`):
                Dictionary of input values.
            max_length (`int`):
                padding to this length.
            return_attention_mask (`bool`, *optional*):
                Set to False to avoid returning attention mask (default: set to model specifics)
        """
        required_input: torch.Tensor = processed_features[self.model_input_names[0]]

        max_length = alt_none(max_length, len(required_input))

        needs_to_be_padded = bool(len(required_input) < max_length)

        if return_attention_mask and "attention_mask" not in processed_features:
            processed_features["attention_mask"] = torch.ones(
                len(required_input), dtype=torch.int32
            )

        if needs_to_be_padded:
            difference = max_length - len(required_input)
            if self.padding_side == "right":
                if return_attention_mask:
                    processed_features["attention_mask"] = F.pad(
                        processed_features["attention_mask"], (0, difference)
                    )
                padding_shape = [0, 0] * (required_input.ndim - 1) + [
                    0,
                    difference,
                ]  # pad first dim
                processed_features[self.model_input_names[0]] = F.pad(
                    required_input,
                    padding_shape,
                    "constant",
                    value=self.padding_value,
                )
            elif self.padding_side == "left":
                if return_attention_mask:
                    processed_features["attention_mask"] = F.pad(
                        processed_features["attention_mask"], (difference, 0)
                    )
                padding_shape = [0, 0] * (required_input.ndim - 1) + [
                    difference,
                    0,
                ]  # pad first dim
                processed_features[self.model_input_names[0]] = F.pad(
                    required_input,
                    padding_shape,
                    "constant",
                    value=self.padding_value,
                )
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))

        return processed_features

    def _truncate(
        self,
        processed_features: Dict,
        max_length: Optional[int] = None,
        truncate: Optional[int] = None,
    ):
        """
        Truncate inputs to predefined length or max length in the batch

        Args:
            processed_features (`Dict[str, torch.Tensor]`):
                Dictionary of input values.
            max_length (`int`, *optional*):
                fix the maximum length of the returned list if `truncate=True`.
            truncate (`bool`, *optional*):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
        """
        if not truncate:
            return processed_features
        elif truncate and max_length is None:
            raise ValueError(
                "When setting ``truncation=True``, make sure that ``max_length`` is defined."
            )
        required_input = processed_features[self.model_input_names[0]]

        needs_to_be_truncated = len(required_input) > max_length

        if needs_to_be_truncated:
            processed_features[self.model_input_names[0]] = processed_features[
                self.model_input_names[0]
            ][:max_length]
            if "attention_mask" in processed_features:
                processed_features["attention_mask"] = processed_features[
                    "attention_mask"
                ][:max_length]

        return processed_features


class KaldiFeatureExtractor(SequenceFeature):
    """
    Kaldi-feature extractor class, inheriting from SequenceFeature.

    Args:
        sampling_rate (Optional[int]):
            The input sampling rate. Defaults to 16000
        feat_conf (Optional[Dict]):
            Feature configuration. Defaults to kaldi-fbank
        scale_bit Optional[int]:
            Whether scale inputs signal to 16/32 bit.
        mean_norm (bool):
            Whether to perform mean normalization on kaldi feature. Defaults to True
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
        mean_norm: bool = True,
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
        extactor = self.extractor
        if rich_feat_info:
            self.feat_conf = extactor.to_dict()

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

    def __call__(
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
            samples (Union[SingleTensor, BatchTensor]):
                Input samples, can be either single tensor audio of shape `[C, T]` or batch of tensors.
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
        >>> extractor = KaldiFeatureExtractor(mean_norm=False)
        >>> signal = torch.sin(torch.arange(16000))
        >>> extractor(signal,sampling_rate=16000)
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
        extractor: LhotseFeat = get_lhotse_feat(**self.feat_conf)
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

    @property
    def extractor(self):
        return get_lhotse_feat(**self.feat_conf)

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


class OfflineKaldiFeatureExtractor(SequenceFeature):
    """
    Offline-Kaldi feature extractor class, just hanle input feat tensors with padding, collates.

    Args:
        mean_norm (bool):
            Whether to perform mean normalization on kaldi feature. Defaults to True
        std_norm (bool):
            Whether to perform standard deviation normalization. Defaults to False
        return_attention_mask (bool):
            Whether to return attention mask.
        padding_value (float):
            Padding value, default is log of `1e-10`.
        feat_conf (Optional[Dict]):
            Feature configuration, for reproduction. Pass a {} will get a kaldi-fbank of 80 dim.
        **kwargs:
            Other arguments.

    Example::

        import torch

        from egrecho.data.features.feature_extractor_audio import OfflineFeatureExtractor

        fbank = (torch.randn(100, 80), torch.randn(75, 80))
        extractor = OfflineFeatureExtractor(mean_norm=False, feat_conf={})
        outs = extractor(fbank, sampling_rate=16000)
        feats, att_mask = outs['input_features'], outs['attention_mask']
        assert tuple(feats.shape) == (2, 100, 80)
        assert (att_mask.sum(dim=-1).tolist() == [100,75])
    """

    # Input names expected by the model
    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        mean_norm: bool = True,
        std_norm: bool = False,
        return_attention_mask: bool = True,
        padding_value: float = LOG_EPSILON,
        feat_conf: Optional[Dict] = None,
        **kwargs,
    ):
        if feat_conf is None or not isinstance(feat_conf, dict):
            raise ConfigurationException(
                "Actually `feat_conf` is useless when processing offline feat, but we must set it to tell us how "
                f"to get the feature from raw wav. Got {feat_conf}. set a dummy {{}} will get a kaldi-fbank of 80 dim."
            )

        self.feat_conf = dict_union(DEFAULT_FEAT_CONF, feat_conf)

        self.mean_norm = mean_norm
        self.std_norm = std_norm
        self.return_attention_mask = return_attention_mask
        online_extractor = self.get_online_extractor()
        sampling_rate, feature_size = (
            online_extractor.sampling_rate,
            online_extractor.feature_dim,
        )

        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )

    def __call__(
        self,
        features: Union[SingleTensor, BatchTensor],
        sampling_rate: Optional[int] = None,
        max_length: Optional[int] = None,
        truncate: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: bool = True,
    ) -> dict:
        """
        Call the feature extractor for feature extraction. `(T, F) -> (B, T, F)`

        Args:
            features (Union[SingleTensor, BatchTensor]):
                Input features, can be either single tensor feature of shape `[T, C]` or batch of tensors.
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

        Example::

            import torch

            from egrecho.data.features.feature_extractor_audio import OfflineFeatureExtractor

            fbank = (torch.randn(100, 80), torch.randn(75, 80))
            extractor = OfflineFeatureExtractor(mean_norm=False, feat_conf={})
            outs = extractor(fbank, sampling_rate=16000)
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
                "The expected feature size, as specified in your Kaldi-extractor (feat_conf), "
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

    def get_online_extractor(self) -> KaldiFeatureExtractor:
        cfg = self.to_dict()
        return KaldiFeatureExtractor.from_dict(cfg)

    @property
    def feature_dim(self) -> int:
        return self.feature_size

    @classmethod
    def available_extractors(self) -> List[str]:
        return FEATURE_EXTRACTORS.keys()


# bc
class OfflineFeatureExtractor(OfflineKaldiFeatureExtractor):
    pass


class RawWavExtractor(SequenceFeature):
    """
    Raw-wav feature extractor class, just hanle input feat tensors with padding, collates.

    Args:
        return_attention_mask (bool):
            Whether to return attention mask.
        padding_value (float):
            Padding value, defaults to `0.`.
        **kwargs:
            Other arguments.
    """

    # Input names expected by the model
    model_input_names = ["input_values", "attention_mask"]

    def __init__(
        self,
        return_attention_mask: bool = True,
        padding_value: float = 0.0,
        **kwargs,
    ):
        # compatible arg
        self.sampling_rate = kwargs.pop("sampling_rate", 16000)

        self.feature_size = None

        super().__init__(
            feature_size=self.feature_size,
            sampling_rate=self.sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )

    def __call__(
        self,
        samples: Union[SingleTensor, BatchTensor],
        sampling_rate: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: bool = True,
    ) -> dict:
        """
        Call the feature extractor for feature extraction. `(C, T)/(T,) -> (B, T, C)/(B, T)`

        Args:
            samples (Union[SingleTensor, BatchTensor]):
                Input samples, can be either single tensor audio of shape `[C, T]` or batch of tensors.
            return_attention_mask (Optional[bool]):
                Whether to return attention mask. if specified, will affect the default set in `__init__`.
            return_tensors (bool):
                If True, output features in batch is a tensor, otherwise list of tensors. Defaults to True.

        Returns:
            dict: A dictionary containing features.

        Example:
        >>> extractor = RawWavExtractor()
        >>> signal = torch.sin(torch.arange(16000))
        >>> extractor(signal,sampling_rate=16000)
        {'input_values': tensor([[[0.0000],
                [0.8415],
                [0.9093],
                ...,
                [0.0102],
                [0.8469],
                [0.9050]]]),
        'attention_mask': tensor([[1, 1, 1,  ..., 1, 1, 1]], dtype=torch.int32)}
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
        samples = [(to_torch_tensor(sample)).transpose(0, -1) for sample in samples]
        batched_samples = {"input_values": samples}
        padded_inputs = self.pad(
            batched_samples, return_attention_mask=return_attention_mask
        )
        if not return_tensors:
            padded_inputs["input_values"] = [sample for sample in padded_inputs]
        return padded_inputs

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
        output.pop("padding_value", None)
        return output


def cmvn_single(
    x: torch.Tensor,
    input_length: Optional[int] = None,
    mean_norm: Optional[bool] = True,
    std_norm: Optional[bool] = False,
    padding_value: float = 0.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Performs mean and variance normalization of the first dimension of input feature.

    Args:
        x (torch.Tensor): Input tensor of shape [T, F].
        input_length (int, optional): Length of input sequence. If None, the whole sequence is used.
        mean_norm (bool, optional): If True, the mean will be normalized.
        std_norm (bool, optional): If True, the standard deviation will be normalized.
        padding_value (float, optional): Value to pad if input_length is shorter than the sequence length.

    Returns:
        torch.Tensor: Normalized tensor.

    Example:
        >>> import torch
        >>> feature = torch.randn([101, 20])
        >>> feature = cmvn_single(feature)
    """

    input_length = x.shape[0] if input_length is None else input_length

    if mean_norm:
        mean = x[:input_length].mean(dim=0)
    if std_norm:
        std = torch.sqrt(x[:input_length].var(dim=0) + eps)
    if mean_norm and std_norm:
        x = (x - mean) / std
    elif mean_norm:
        x = x - mean
    elif std_norm:
        x = x / std
    if input_length < x.shape[0] and (mean_norm or std_norm):
        x[input_length:] = padding_value

    return x


def cmvn_utts(
    input_features,
    attention_mask: Optional[torch.Tensor] = None,
    lengths: Optional[torch.Tensor] = None,
    mean_norm: Optional[bool] = True,
    std_norm: Optional[bool] = False,
    eps: float = 1e-7,
):
    """Performs mean and variance normalization of the second dimension of batch input features.

    Args:
        input_features (torch.Tensor): Input tensor of shape [B, T, F].
        attention_mask (torch.Tensor): Mask of shape [B, T, F].
        lengths (torch.Tensor): Length [B,] of input sequence. Note that can't set with ``attention_mask``/
        mean_norm (bool, optional): If True, the mean will be normalized.
        std_norm (bool, optional): If True, the standard deviation will be normalized.
        padding_value (float, optional): Value to pad if input_length is shorter than the sequence length.

    Returns:
        torch.Tensor: Normalized tensor.

    Example:
        >>> import torch
        >>> inputs = torch.randn([10, 101, 20])
        >>> inp_len = torch.zeros((10,)).uniform_(0, 1) * inputs.shape[1]
        >>> features = cmvn_utts(inputs, inp_len)

    """
    if not mean_norm and not std_norm:
        return input_features

    bsz = len(input_features)
    input_is_list = isinstance(input_features, (list, tuple))
    if (int(attention_mask is None) + int(lengths is None)) == 0:
        raise ValueError(f"Set both attention_mask and lengths is ilegal.")
    elif attention_mask is not None:
        if isinstance(attention_mask, (list, tuple)):
            lengths = [attention_mask[i].sum(-1).long() for i in range(bsz)]
        else:
            lengths = attention_mask.sum(-1).long()
    elif lengths is not None:
        lengths = (
            [int(le) for le in lengths]
            if isinstance(lengths, (list, tuple))
            else lengths.long()
        )
    else:
        lengths = (
            [fea.shape[0] for fea in input_features]
            if input_is_list
            else [input_features.shape[1]] * bsz
        )

    assert len(lengths) == bsz, len(lengths)

    outs = []
    for snt_id in range(bsz):

        outs.append(
            cmvn_single(
                input_features[snt_id],
                input_length=lengths[snt_id],
                mean_norm=mean_norm,
                std_norm=std_norm,
                eps=eps,
            )
        )
    if input_is_list:
        return outs
    else:
        # tensor
        return torch.stack(outs, dim=0)
