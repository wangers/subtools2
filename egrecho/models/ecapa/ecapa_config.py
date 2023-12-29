# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-08)

from dataclasses import dataclass, field

from typing_extensions import Literal, Optional

from egrecho.core.config import DataclassConfig

default_pooling_params = {
    "num_q": 1,
    "num_head": 1,
    "time_attention": True,
    "hidden_size": 128,
    "stddev": True,
}


@dataclass
class EcapaConfig(DataclassConfig):
    """
    Configuration class for the Ecapa.

    Args:
        inputs_dim (int, optional):
            The dimension of the input feature. Default is 80.
        num_targets (int, optional):
            The number of target classes. Default is 2.
        channels (int, optional):
            The number of channels in the model. Default is 512.
        embd_dim (int, optional):
            The embedding dimension. Default is 192.
        mfa_dim (int, optional):
            The dimension of the MFA layer. Default is 1536.
        pooling_method (str, optional):
            The pooling method used in the model. Default is 'mqmhasp'.
        pooling_params (Dict, optional): Pooling parameters.
            Default is a predefined dictionary of `asp`.
        embd_layer_num (Literal[1, 2], optional):
            The number of embedding layers. Default is 1.
        post_norm (bool, optional):
            Whether to apply post-normalization. Default is False.
    """

    inputs_dim: int = 80
    channels: int = 512
    embd_dim: int = 192
    mfa_dim: int = 1536
    pooling_params: dict = field(default_factory=lambda: default_pooling_params)
    embd_layer_num: Literal[1, 2] = 1
    post_norm: bool = True

    def __post_init__(self):
        self.pooling_params = {**default_pooling_params, **self.pooling_params}


@dataclass
class EcapaSVConfig(EcapaConfig):
    """
    Configuration class for the Ecapa with Classification task.

    Args:
        num_classes:
            Need to be set to label number. Defaults to 2.
        classifier_str (str, optional):
            Margin classifier. Default is aam.
        classifier_params (dict, optinal):
            other kwargs passing to margin classifier.
            sub_k: sub center. Default is 1.
            do_topk: wheter do hard sample margin.
    """

    num_classes: int = 2
    classifier_str: Optional[Literal["linear", "am", "aam"]] = "aam"
    classifier_params: dict = field(
        default_factory=lambda: {"sub_k": 1, "do_topk": False}
    )

    def __post_init__(self):
        super().__post_init__()
