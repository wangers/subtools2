# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2024-04)

from dataclasses import dataclass

from typing_extensions import Literal, Optional

from egrecho.core.config import DataclassConfig
from egrecho.utils.misc import ConfigurationException


@dataclass
class EncodecConfig:
    """Defaut 24k encodec.

    Args:
        sampling_rate (int, defaults to 24k):
            24/48k encodec model.
        frame_rate (int, defaults to 75):
            token number of per second.
        codebook_size (`int`, *optional*, defaults to 1024):
            Encodec size of the codebook. Defines the number of different audio tokens that can be represented by the
            `inputs_values` passed.
        num_codebooks (`int`, *optional*, defaults to 8):
            Number of codebooks (residual quantized vectors).
    """

    sampling_rate: int = 24_000
    num_codebooks: int = 8
    frame_rate: int = 75
    codebook_size: int = 1024

    @property
    def target_bandwidths(self) -> int:
        return round((self.frame_rate * 10) * self.num_codebooks / 1000, 0)


@dataclass
class ValleConfig(DataclassConfig):
    """
    Configuration class for the Valle submodel (nar & ar).

    Args:
        vocab_size (`int`, *optional*, defaults to 512):
            Vocabulary size of the model. Defines the number of different text tokens that can be represented by the
            `inputs_ids` passed.
        encodec_config (`dataclass`):
            Some specify conifg of encodec.
        rope (`bool`, *optional*, defaults to `False`):
            Whether apply rope
        num_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the given sub-model.
        num_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer architecture.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the architecture.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        ffn_type (`str`, *optional*, defaults to `"normal"`):
            GLU FFN or tranditional FFN.
        hidden_act (`str`, *optional*, defaults to `"swish"`):
            The non-linear activation function (function or string) in the decoder.
        pad_text_token_id (`int`, *optional*, defaults to 1):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        bias (`bool`, *optional*, defaults to `True`):
            Whether or not to use bias in the linear layers and layer norm layers.
        norm_type (`str`, *optional*, defaults to `ln`):
            Layer norm or RMS norm.
        use_sdpa (`bool`, *optional*, defaults to `True`):
            Whether use torch scale dot attention which support flash attention.
        prefix_mode (`str`, *optional*, defaults to `starter`):
            Use prefix audio codebooks as prompt when training NAR sub-model.
            `starter` means random seg of start of inputs, `exter` means provide manually.
    """

    vocab_size: int = 512
    encodec_config: Optional[EncodecConfig] = None
    rope: bool = False
    num_layers: int = 12
    num_heads: int = 16
    hidden_size: int = 1024
    dropout: float = 0.1
    ffn_type: Literal["normal", "gated"] = "normal"
    hidden_act: str = "swish"
    bias: bool = True
    norm_type: Literal["rms", "ln"] = "ln"
    norm_eps: Optional[float] = None
    use_sdpa: bool = True
    prefix_mode: Literal["starter", "exter"] = "starter"
    pad_text_token_id: int = 0

    def __post_init__(self):
        assert self.prefix_mode in ["starter", "exter"], self.prefix_mode
        self.encodec_config = self.encodec_config or EncodecConfig()
        self.norm_eps = self.norm_eps or (1e-5 if self.norm_type == "ln" else 1e-6)

    @property
    def codebook_size(self):
        return self.encodec_config.codebook_size

    @property
    def num_codebooks(self):
        return self.encodec_config.num_codebooks

    @property
    def frame_rate(self):
        return self.encodec_config.frame_rate


@dataclass
class ValleModelConfig(ValleConfig):
    """
    Configuration class for the Valle model.

    Args:
        has_ar (`bool`, *optional*, defaults to True):
            Whether contains ar submodel.
        has_nar (`bool`, *optional*, defaults to True):
            Whether contains nar submodel.

    """

    has_ar: bool = True
    has_nar: bool = True

    def __post_init__(self):
        super().__post_init__()
        if not (self.has_ar or self.has_nar):
            raise ConfigurationException(
                "Invalid config as neither ar & nar are set to exist."
            )
