# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-08)

from dataclasses import dataclass, field

from typing_extensions import Literal, Optional

from egrecho.core.config import DataclassConfig


@dataclass
class CamPPConfig(DataclassConfig):
    """
    Configuration class for the cam++.

    Args:
        inputs_dim (int, optional):
            The dimension of the input feature. Defaults to 80.
        embd_dim (int, optional):
            The embedding dimension. Defaults to 512.
        init_channels (int, optional):
            Before D-TDNN blocks, transform fcm outdim to init_channels. Defaults to 128.
        growth_rate (int, optional):
            out dimenstion of each inside D-TDNN layer.
        bn_size:
            Multiplier of D-TDNN layer's out dim, which will apply cam attention,
            i.e., `bn_size*growth_rate`. Defualts to 4.
        memory_efficient (bool, optional):
            Whether use gradient checkpointing when training. Defaults to True.
    """

    inputs_dim: int = 80
    embd_dim: int = 512
    init_channels: int = 128
    growth_rate: int = 32
    bn_size: int = 4
    memory_efficient: bool = True


@dataclass
class CamPPSVConfig(CamPPConfig):
    """
    Configuration class for the cam++ with Classification task.

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
