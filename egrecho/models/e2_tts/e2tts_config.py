# -*- coding:utf-8 -*-
# (Author: Leo 2024-08)
from dataclasses import dataclass

from typing_extensions import Optional, Tuple, Union

from egrecho.core.config import DataclassConfig, normalize_dict
from egrecho.utils.misc import ConfigurationException


@dataclass
class MSizes(DataclassConfig):
    """Control scale size.

    Args:
        num_layers:
            Number of hidden layers in the given sub-model.
        num_heads:
            Number of attention heads for each attention layer in the Transformer architecture.
        d_head:
            Dimensionality of qkv head dim.
        hidden_size:
            Dimensionality of the hidden_size in the architecture.
    """

    num_layers: int
    num_heads: int
    d_head: int
    hidden_size: int


DiT_MAP = {
    "dit-xl": MSizes(28, 16, 72, 1152),
    "dit-l": MSizes(24, 16, 64, 1024),
    "dit-b": MSizes(12, 12, 64, 768),
    "dit-s": MSizes(12, 6, 64, 384),
}


@dataclass
class ModelConfig(DataclassConfig):
    """
    Configuration class for the backbone model.

    Args:
        vocab_size (`int`, *optional*, defaults to 512):
            Vocabulary size of the model. Defines the number of different text tokens that can be represented by the
            `inputs_ids` passed.
        inputs_dim (`int`, *optional*, defaults to 100):
            inp dim.
        backbone:
            Registered backbone names (dit-s|dit-b|dit-l|dit-xl).
        backbone_sizes:
            Override sizes (hidden_size, num_heads, num_layers)
        num_key_value_heads (`int`, *optional*, defaults to 16):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. If it is not specified, will default to
            `num_attention_heads`.
        attn_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for attn sim score.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        ntk_max_position (`int`, *optional*, defaults to 8192):
            The maximum sequence length that this model might ever be used with.
        qkv_bias (`bool`, *optional*, defaults to `False`):
            Whether or not to use bias in the qkv project layers.
        qk_norm (`bool`, *optional*, defaults to `True`):
            Norm qk as dit to stable training.
        sandwish_norm (`bool`, *optional*, defaults to `False`):
            If true, apply a additional post norm.
        use_sdpa (`bool`, *optional*, defaults to `True`):
            Whether use torch scale dot attention which support flash attention.
        softclamp_logits:
            whether applying tanh softcapping
        softclamp_logits_val (`float`, *optional*, defaults to 50.0):
            scaling factor when applying tanh softcapping on the attention scores.
        cond_text_net:
            whether use a subnet to transform text embd every block, if False, concat cond input text as paper.
        pad_text_token_id:
            pad tok id
        pad_feats_val:
            pad feature val
    """

    vocab_size: int = 256
    inputs_dim: int = 100
    backbone: str = "dit-s"
    unet_mode: bool = False
    backbone_sizes: Optional[MSizes] = None
    num_key_value_heads: Optional[int] = None
    attn_dropout: float = 0
    dropout: float = 0.1
    ntk_max_position: int = 8192
    qkv_bias: bool = False
    qk_norm: bool = True
    sandwish_norm: bool = False
    norm_eps: float = 1e-6
    use_sdpa: bool = True
    softclamp_logits: bool = False
    softclamp_logits_val: Optional[float] = 50.0
    cond_text_net: bool = False
    pad_text_token_id: int = 0
    pad_feats_val: float = 0.0

    def __post_init__(self):

        self.backbone = self.backbone.lower()
        register_sizes = DiT_MAP.get(self.backbone, {}) if self.backbone else {}
        self.backbone_sizes = self.backbone_sizes or {}
        if not (register_sizes or self.backbone_sizes):
            raise ConfigurationException(
                f"Neither a valid backbone name nor backbone_sizes is configured."
            )
        self.backbone_sizes = MSizes.from_config(
            register_sizes, **normalize_dict(self.backbone_sizes)
        )
        self.num_key_value_heads = self.num_key_value_heads or self.num_heads
        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_heads must be divisible by num_key_value_heads (got `num_heads`: {self.num_heads}"
                f" and `num_key_value_heads`: {self.num_key_value_heads})."
            )

    @property
    def num_layers(self):
        return self.backbone_sizes.num_layers

    @property
    def num_heads(self):
        return self.backbone_sizes.num_heads

    @property
    def hidden_size(self):
        return self.backbone_sizes.hidden_size

    @property
    def d_head(self):
        return self.backbone_sizes.d_head


@dataclass
class E2TTSConfig(ModelConfig):
    """
    Configuration class for the E2TTS.

    Args:
        with_fourier_features (`bool`, *optional*, defaults to `True`):
            simple fourier features to enrich inputs.
        frac_lengths_mask:
            mask percentage range for training
        sigma:
            ode sigma
        cond_inp_add:
            add/concat condition inp
        cond_dropout (`float`, *optional*, defaults to `0.2`):
           probs of condition dropout (e.g., z and x_ctx in the paper).

    """

    with_fourier_features: bool = False
    frac_lengths_mask: Tuple[float, float] = (0.7, 1.0)
    cond_dropout: float = 0.2
    cond_inp_add: bool = False
    sigma: float = 0.0

    def __post_init__(self):
        super().__post_init__()


@dataclass
class DurPredConfig(ModelConfig):
    """
    Configuration class for the DurationPredictor.
    """

    def __post_init__(self):
        super().__post_init__()
