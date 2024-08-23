# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2024-04)
"""
VALLE implementation.

Refs:
    paper: Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers.
        http://arxiv.org/abs/2301.02111
    repo: http://github.com/lifeiteng/vall-e
"""
import contextlib
import math
from typing import Any, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from egrecho.core.model_base import ModelBase
from egrecho.models.valle.valle_config import ValleConfig
from egrecho.nn.activation import Nonlinearity
from egrecho.utils.cuda_utils import avoid_float16_autocast_context
from egrecho.utils.mask import (
    make_causal_mask,
    make_non_pad_mask,
    prepare_4d_attention_mask,
)

MaskCache = torch.Tensor
PECache = torch.Tensor
KVCache = Tuple[torch.Tensor, torch.Tensor]


def apply_rope(x: torch.Tensor, rotary_pe: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x (`torch.Tensor`): The tensor of shape [bsz, n_head, T, d_model//n_head].
        rotary_emed (`torch.Tensor`): Precomputed rotary of shape [b_rope, T, d_model//n_head//2, 2].
    """

    assert x.shape[2] == rotary_pe.shape[1]

    xshaped = x.reshape(*x.shape[:-1], -1, 2)  # (bsz, n_head, T, d_model//n_head//2, 2)
    rotary_pe = rotary_pe.unsqueeze(1).type_as(
        x
    )  # b_rope, 1, T, d_model//n_head//2, 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rotary_pe[..., 0] - xshaped[..., 1] * rotary_pe[..., 1],
            xshaped[..., 1] * rotary_pe[..., 0] + xshaped[..., 0] * rotary_pe[..., 1],
        ],
        -1,
    )
    return x_out2.flatten(3)


class SineEmbedding(nn.Module):
    """
    Sine Positional encoding.

        - ``PE(pos, 2i)   = sin(pos/(10000**(2i/dmodel)))``
        - ``PE(pos, 2i+1) = cos(pos/(10000**(2i/dmodel)))``

    Args:
        d_model: embedding dim
        dropout: dropout rate
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.0,
        scale: bool = False,
        alpha: bool = False,
        device=None,
    ):
        super().__init__()
        assert d_model % 2 == 0
        self.d_model = d_model
        self.x_scale = math.sqrt(d_model) if scale else 1.0
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=alpha)
        self.dropout = torch.nn.Dropout(p=dropout)
        inv_freq = 1.0 / (
            10000.0
            ** (
                torch.arange(0, self.d_model, 2, dtype=torch.int64).float().to(device)
                / self.d_model
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def _forward(self, x, position_ids):

        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            cos = freqs.cos()
            sin = freqs.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, sequence_length, ...)
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings.

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
            torch.Tensor: position encodings of x of shape (bsz, time, ...).
        """
        cos, sin = self._forward(x, position_ids)

        # (bsz, seq_len, dim//2, 2)
        pos_emb = torch.stack([sin, cos], dim=-1)
        # (bsz, seq_len, dim)
        pos_emb = pos_emb.view(*pos_emb.shape[:-2], -1)
        output = x.unsqueeze(-1) if x.ndim == 2 else x
        output = output * self.x_scale + self.alpha * pos_emb
        return self.dropout(output), pos_emb


class RotaryEmbedding(nn.Module):
    """Rotary positional encoding module.
    RoFormer: Enhanced Transformer with Rotary Position Embedding.

    Args:
        d_model (int): Embedding dimension.
        n_head (int): head num.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        device=None,
    ):
        super().__init__()
        assert (d_model % n_head) % 2 == 0
        self.d_model = d_model
        self.d_head = d_model // n_head
        inv_freq = 1.0 / (
            10000.0
            ** (
                torch.arange(0, self.d_head, 2, dtype=torch.int64).float().to(device)
                / self.d_head
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def _forward(self, x: torch.Tensor, position_ids: torch.Tensor):

        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            cos = freqs.cos()
            sin = freqs.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, sequence_length, ...)
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings.

        Returns:
            torch.Tensor: inputs tensor without encoding, unify interface with SinePositionalEmbedding.
            torch.Tensor: position encodings of x of shape (bsz, time, ...).
        """
        cos, sin = self._forward(x, position_ids)
        # (bsz, seq_len, dim//n_head//2, 2)
        pos_emb = torch.stack([cos, sin], dim=-1)
        output = x.unsqueeze(-1) if x.ndim == 2 else x
        return output, pos_emb


class TokenEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        self.dropout = torch.nn.Dropout(p=dropout)
        self.word_embeddings = nn.Embedding(self.vocab_size, self.d_model)

    @property
    def weight(self) -> torch.Tensor:
        return self.word_embeddings.weight

    def embedding(self, index: int) -> torch.Tensor:
        return self.word_embeddings.weight[index : index + 1]

    def forward(self, x: torch.Tensor):
        X = self.word_embeddings(x)
        X = self.dropout(X)

        return X


def norm_fac(config: ValleConfig, ada: bool = False):
    if ada:
        if config.norm_type == "rms":
            return AdaptiveRMSNorm(
                config.hidden_size,
                eps=config.norm_eps,
            )
        else:
            return AdaptiveLayerNorm(
                config.hidden_size,
                bias=config.bias,
                norm_type=config.norm_type,
                eps=config.norm_eps,
            )
    return (
        RMSNorm(config.hidden_size, eps=config.norm_eps)
        if config.norm_type == "rms"
        else LayerNorm(config.hidden_size, bias=config.bias, eps=config.norm_eps)
    )


class AdaptiveLayerNorm(nn.Module):
    r"""Adaptive Layer Normalization"""

    def __init__(
        self,
        d_model: int,
        bias: bool = True,
        norm_type: Literal["rms", "ln"] = "ln",
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.project_layer = nn.Linear(d_model, 2 * d_model, bias=bias)
        self.norm = (
            RMSNorm(d_model, eps=eps)
            if norm_type == "rms"
            else LayerNorm(d_model, bias=bias, eps=eps)
        )
        self.d_model = d_model

    def forward(self, input: Tensor, embedding: Tensor) -> Tensor:

        weight, bias = torch.split(
            self.project_layer(embedding),
            split_size_or_sections=self.d_model,
            dim=-1,
        )
        out = weight * self.norm(input) + bias
        return out


class AdaptiveRMSNorm(nn.Module):
    r"""Adaptive RMS Normalization"""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self._has_post_initialized = True  # skip global init
        self.eps = eps
        self.project_layer = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.project_layer.weight)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, input: Tensor, embedding: Tensor) -> Tensor:
        x = self._norm(input.float())
        weight = self.project_layer(embedding).float()

        x = (weight + 1.0) * x
        return x.type_as(input)


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # self.weight = nn.Parameter(torch.ones(d_model))
        self.weight = nn.Parameter(torch.zeros(d_model))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x, embedding: Tensor = torch.empty(0)):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False."""

    def __init__(self, d_model, bias=True, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None
        self.eps = eps

    def forward(self, input, embedding: Tensor = torch.empty(0)):
        return F.layer_norm(
            input, self.weight.shape, self.weight, self.bias, eps=self.eps
        )


class ValleAttention(nn.Module):

    sdpa_32bit = False

    def __init__(self, config: ValleConfig):
        super().__init__()

        # regularization
        self.dropout = config.dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads

        if config.hidden_size % config.num_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.spda_att = config.use_sdpa and hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )
        if not self.spda_att:
            msg = "Using slow attention. Flash Attention requires use_sdpa=True and PyTorch >= 2.0"
            try:
                from egrecho.utils.logging import get_logger

                log = get_logger()
                log.warning_once(msg, ranks=0)
            except Exception as exc:  # noqa
                print(f"WARNING: {msg}")
        # key, query, value projections for all heads, but in a batch
        self.att_proj = nn.Linear(
            config.hidden_size, 3 * config.hidden_size, bias=config.bias
        )
        # output projection
        self.out_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.bias
        )

    # Copied from transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoSelfAttention._split_heads
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """

        # re-assemble all head outputs side by side
        # (batch, num_heads, seq_len, attn_head_size) -> (batch, seq_len, num_heads*attn_head_size)
        tensor = tensor.transpose(1, 2).contiguous()
        tensor = tensor.view(tensor.size()[:-2] + (num_heads * attn_head_size,))

        return tensor

    def _attn(self, query, key, value, attention_mask=None):

        if self.spda_att:
            # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
            # Reference: https://github.com/pytorch/pytorch/issues/112577.
            if query.device.type == "cuda" and attention_mask is not None:
                query = query.contiguous()
                key = key.contiguous()
                value = value.contiguous()

                attention_mask = _cast_attn_bias(attention_mask, query.dtype)

            ctx = (
                avoid_float16_autocast_context()
                if self.sdpa_32bit
                else contextlib.nullcontext()
            )
            with ctx:
                attn_output = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    dropout_p=self.dropout if self.training else 0,
                )
            return attn_output
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
            self.head_dim
        )
        if attention_mask is not None:
            # Apply the attention mask
            attention_mask = _cast_attn_bias(attention_mask, attn_weights.dtype)
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.max(
            attn_weights,
            torch.tensor(
                torch.finfo(attn_weights.dtype).min, device=attn_weights.device
            ),
        )

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights.float(), dim=-1).to(
            query.dtype
        )

        attn_weights = self.attn_dropout(attn_weights)

        # (batch, num_heads, seq_len, seq_len) x (batch, num_heads, seq_len, attn_head_size)
        # -> (batch, num_heads, seq_len, attn_head_size)
        attn_output = torch.matmul(attn_weights, value)

        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[KVCache] = None,
        rotary_pe: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ):
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query, key, value = self.att_proj(hidden_states).split(self.embed_dim, dim=2)

        # (bsz, head, seq, dim)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        if rotary_pe is not None:
            key = apply_rope(key, rotary_pe=rotary_pe)
            query = apply_rope(query, rotary_pe=rotary_pe)
        if past_key_value is not None:
            past_key = past_key_value[0]
            past_value = past_key_value[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output = self._attn(query, key, value, attention_mask=attention_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)

        return outputs


def fnn_fac(config: ValleConfig):
    ffn = GatedFFN if config.ffn_type == "gated" else DenseFFN
    return ffn(
        config.hidden_size,
        4 * config.hidden_size,
        dropout_rate=config.dropout,
        activation_type=config.hidden_act,
        bias=config.bias,
    )


class DenseFFN(torch.nn.Module):
    """Positionwise feed forward

    Args:
        dim: input dimenstion
        hidden_dim: number of hidden units
        dropout_rate: dropout rate
    """

    def __init__(
        self,
        dim,
        hidden_dim,
        dropout_rate=0.1,
        activation_type="swish",
        bias=True,
    ):
        super().__init__()
        self.w_1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.activation = Nonlinearity(activation_type)

    def forward(self, x):
        x = self.w_1(x)
        return self.dropout1(self.out_proj(self.dropout(self.activation(x))))


class GatedFFN(nn.Module):
    """GLU feed forward

    Args:
        dim: input dimenstion
        hidden_dim: number of hidden units
        dropout_rate: dropout rate
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout_rate=0.1,
        activation_type="swish",
        bias=True,
        multiple_of: int = 256,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.dropout = nn.Dropout(dropout_rate)
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        self.activation = Nonlinearity(activation_type)

    def forward(self, x):
        return self.out_proj(self.dropout(self.activation(self.w1(x)) * self.w3(x)))


class ArBlock(nn.Module):
    """Causual Valle Transformer Block"""

    def __init__(self, config: ValleConfig, layer_idx):
        super().__init__()

        self.norm_1 = norm_fac(config)
        self.attn = ValleAttention(config)
        self.norm_2 = norm_fac(config)
        self.mlp = fnn_fac(config)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        rotary_pe: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        **kwargs,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, 1, query_sequence_length, key_sequence_length)`.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            rotary_pe (`torch.Tensor`):
                If not None, apply precomputed rotary of shape [bsz, T, d_model//n_head//2, 2].
        """
        attn_output, new_key_value = self.attn(
            self.norm_1(hidden_states),
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            rotary_pe=rotary_pe,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(self.norm_2(hidden_states))
        outputs = (hidden_states,)

        if use_cache:
            outputs += (new_key_value,)

        return outputs


class NarBlock(nn.Module):
    """Non Causual Valle Transformer Block"""

    def __init__(self, config: ValleConfig, layer_idx):
        super().__init__()

        self.norm_1 = norm_fac(config, ada=config.ada_norm)
        self.attn = ValleAttention(config)
        self.post_att_norm = norm_fac(config)
        self.attn.sdpa_32bit = config.nar_sdpa_32bit
        self.norm_2 = norm_fac(config, ada=config.ada_norm)
        self.mlp = fnn_fac(config)
        self.post_mlp_norm = norm_fac(config)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pe: Optional[torch.Tensor] = None,
        stage_embedding: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, 1, query_sequence_length, key_sequence_length)`.
            rotary_pe (`torch.Tensor`):
                If not None, apply precomputed rotary of shape [bsz, T, d_model//n_head//2, 2].
        """
        attn_output, _ = self.attn(
            self.norm_1(hidden_states, stage_embedding),
            attention_mask=attention_mask,
            rotary_pe=rotary_pe,
        )

        attn_output = self.post_att_norm(attn_output)

        hidden_states = hidden_states + attn_output
        outputs = hidden_states + self.post_mlp_norm(
            self.mlp(self.norm_2(hidden_states, stage_embedding))
        )
        return outputs


class ValleBase(ModelBase):
    """
    Abstract base module for valle.
    """

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class Decoder(ValleBase):
    def __init__(
        self,
        config: ValleConfig,
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.rope = config.rope

        self.text_embeddings = TokenEmbedding(config.hidden_size, config.vocab_size)
        if self.is_ar:
            self.audio_embeddings = TokenEmbedding(
                config.hidden_size, config.codebook_size + 1
            )
        else:
            self.audio_embeddings = nn.ModuleList(
                [
                    TokenEmbedding(config.hidden_size, config.codebook_size + 1)
                    for _ in range(config.num_codebooks)
                ]
            )
        if self.rope:
            self.text_position = RotaryEmbedding(
                config.hidden_size,
                config.num_heads,
            )
            self.audio_position = RotaryEmbedding(
                config.hidden_size,
                config.num_heads,
            )
        else:
            self.text_position = SineEmbedding(
                config.hidden_size,
                dropout=config.dropout if self.is_ar else 0.0,
                scale=False,
                alpha=self.is_ar,
            )
            self.audio_position = SineEmbedding(
                config.hidden_size,
                dropout=config.dropout,
                scale=False,
                alpha=self.is_ar,
            )
        block_cls = ArBlock if self.is_ar else NarBlock
        self.layers = nn.ModuleList(
            [block_cls(config, layer_idx) for layer_idx in range(config.num_layers)]
        )
        self.norm_f = norm_fac(config)

    @property
    def is_ar(self) -> bool:
        raise NotImplementedError


class ArDecoder(Decoder):
    def __init__(
        self,
        config: ValleConfig,
    ):
        super().__init__(config)
        self.lm_head = nn.Linear(
            config.hidden_size, config.codebook_size + 1, bias=False
        )
        # apply global weight init
        self.post_init()
        # apply special scaled init to the residual projections, GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("out_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers)
                )

    @property
    def is_ar(self) -> bool:
        return True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Tuple[KVCache] = None,
        input_pos: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length, number_of_codebooks)`, *optional*):
                It will lately choose the first of discret code embeddings `(batch_size, sequence_length, 1)`.
            attention_mask (`torch.LongTensor` of shape `(batch_size, kv_len)`, *optional*):
                Default behavior: generate a tensor that ignores pad tokens in `input_ids`. Causal mask will also
                be used by default.
                If `past_key_values` is used, `decoder_attention_mask` needs to contain the masking strategy that was used for
                `past_key_values`. In other words, the `decoder_attention_mask` always has to have the length:
                `len(past_key_values) + len(decoder_input_ids)`

            text_input_ids (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            text_attention_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model.
                Can be used to speed up sequential decoding.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length, num_codebooks)`, *optional*):
                Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
                `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
                are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
            input_pos (`bool`, *optional*):
                If not None, `past_key_values` key value states are returned, streaming mode and can be used to speed up decoding (see
                `past_key_values`).
        """
        use_cache = input_pos is not None
        if input_ids.ndim == 2:
            pass
        elif input_ids.ndim == 3:
            input_ids = input_ids[:, :, 0]
        else:
            raise ValueError(
                f"Invalid shape of decoder_input_ids of {input_ids.ndim}. "
                f"It should be `(batch_size, sequence_length, n)` or `(batch_size, sequence_length)`."
            )

        start_pos = input_pos or 0

        # tok embedding
        input_embeds = self.audio_embeddings(input_ids)

        device, dtype = input_embeds.device, input_embeds.dtype
        input_bs_seq = input_ids.shape[:2]
        decoder_position_ids = torch.arange(
            start_pos, start_pos + input_bs_seq[1], device=device
        ).unsqueeze(0)
        input_embeds, inputs_pe = self.audio_position(
            input_embeds, decoder_position_ids
        )
        if not self.rope:
            inputs_pe = None

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if not start_pos:
            text_embeds = self.text_embeddings(text_input_ids)
            text_position_ids = torch.arange(
                0, text_input_ids.shape[1], device=device
            ).unsqueeze(0)
            text_embeds, text_inputs_pe = self.text_position(
                text_embeds, text_position_ids
            )
            input_embeds = torch.cat([text_embeds, input_embeds], dim=1)

            if not self.rope:
                text_inputs_pe = None
            else:
                inputs_pe = torch.cat([text_inputs_pe, inputs_pe], dim=1)
            if attention_mask is None:
                attention_mask = torch.ones(input_bs_seq, device=device)
            if text_attention_mask is None:
                text_attention_mask = torch.ones(
                    text_input_ids.shape[:2], device=device
                )

            # (bsz, seq_len + text_sequence_length)
            attention_mask = torch.cat([text_attention_mask, attention_mask], dim=1)

        else:
            if attention_mask is None:
                raise ValueError("Requires attention_mask in streaming decoding mode.")

            expected_shape = (input_bs_seq[0], past_key_values_length + input_bs_seq[1])
            if tuple(attention_mask.shape) != expected_shape:
                raise ValueError(
                    f"Incorrect 2D attention_mask shape: {tuple(attention_mask.shape)}; expected: {expected_shape}."
                )
        # (bsz, seq_len + past_key_values_length)
        cache_attention_mask = attention_mask

        prefix_casual_length = (
            text_input_ids.shape[1] if not start_pos else past_key_values_length
        )
        casual_decoder_mask = make_causal_mask(
            input_ids,
            dtype,
            past_key_values_length=prefix_casual_length,
        )

        # (bsz, 1, seq_len, seq_len + past_key_values_length)
        attention_4d_mask = prepare_4d_attention_mask(
            attention_mask, dtype, tgt_len=input_embeds.shape[1]
        )

        if not start_pos:
            attention_4d_mask = attention_4d_mask.clone()  # copy in-place edit
            attention_4d_mask[
                :, :, :prefix_casual_length, prefix_casual_length:
            ] = torch.full_like(
                attention_4d_mask[:, :, :prefix_casual_length, prefix_casual_length:],
                fill_value=torch.finfo(dtype).min,
            )
            attention_4d_mask[
                :, :, prefix_casual_length:
            ] = casual_decoder_mask.masked_fill(
                attention_4d_mask[:, :, prefix_casual_length:].bool(),
                torch.finfo(dtype).min,
            )

        else:
            attention_4d_mask = casual_decoder_mask.masked_fill(
                attention_4d_mask.bool(), torch.finfo(dtype).min
            )

        # embed positions
        hidden_states = input_embeds

        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_4d_mask,
                past_key_value=past_key_value,
                rotary_pe=inputs_pe,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[1],)
        hidden_states = self.norm_f(hidden_states)
        next_cache = next_decoder_cache if use_cache else None
        cache_attention_mask = cache_attention_mask if use_cache else None

        # (B, T, codebook_size + 1)
        logits = self.lm_head(hidden_states[:, -input_bs_seq[1] :])
        return tuple(
            v
            for v in [logits, hidden_states, next_cache, cache_attention_mask]
            if v is not None
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        text_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        phn_dur: float = 0.22,
        top_k: int = -100,
        temperature: float = 0.9,
        top_p: float = 1.0,
    ):
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length, number_of_codebooks)`, *optional*):
                It will lately choose the first of discret code embeddings `(batch_size, sequence_length, 1)`.
            text_input_ids (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention, Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

            text_attention_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`.
            phn_dur (`float`, *optional*, default to 0.22):
                Duration per phn, relevant to approximate the maximum length of the generated audio code sequence.
            topk (`int`, *optional*):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
                if negative, ignore topk filter. Default to -100.
            temperature (`float`, *optional*):
                Temperature value for controlling randomness in sampling. Defaults to 1.0.
            top_p (`float`, *optional*):
                Top-p probability threshold for nucleus sampling. Defaults to 1.
        """
        bsz = text_input_ids.shape[0]
        device = input_ids.device
        src_code_3d = input_ids.ndim == 3

        if input_ids.ndim == 2:
            pass
        elif src_code_3d:
            input_ids = input_ids[:, :, 0]
        else:
            raise ValueError(
                f"Invalid shape of decoder_input_ids of {input_ids.ndim}. "
                f"It should be `(batch_size, sequence_length, n)` or `(batch_size, sequence_length)`."
            )
        prompts_lens = (
            torch.full((bsz,), input_ids.shape[1], dtype=torch.long, device=device)
            if attention_mask is None
            else attention_mask.sum(-1)
        )
        txt_lens = (
            torch.full((bsz,), text_input_ids.shape[1], dtype=torch.long, device=device)
            if text_attention_mask is None
            else text_attention_mask.sum(-1)
        )
        max_lens = (self.config.frame_rate * phn_dur * txt_lens).max()
        min_prompt_len = int(prompts_lens.min().item())
        # generate at least 1 frame
        audio_total_len = int(max(max_lens.item(), input_ids.shape[1] + 1))

        pad_id = self.config.codebook_size

        input_ids, _ = padding_codes(
            input_ids, attention_mask=attention_mask, pad_value=pad_id
        )
        tokens = torch.full(
            (bsz, audio_total_len), pad_id, dtype=torch.long, device=device
        )
        tokens[:, : input_ids.shape[1]] = input_ids

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=device)
        eos_idx = torch.full_like(
            eos_reached, fill_value=min_prompt_len, dtype=torch.int32, device=device
        )

        stop_id = pad_id

        # those generated is True
        input_audio_mask = tokens != pad_id

        kv_cache = None
        cache_attention_mask = attention_mask

        for cur_pos in range(min_prompt_len, audio_total_len):

            # expand att mask cache
            if prev_pos > 0:
                cache_attention_mask = torch.cat(
                    [
                        cache_attention_mask,
                        (~eos_reached[..., None]).to(dtype=attention_mask.dtype),
                    ],
                    dim=1,
                )

            logits, _, kv_cache, cache_attention_mask = self.forward(
                tokens[:, prev_pos:cur_pos],
                attention_mask=cache_attention_mask,
                text_input_ids=text_input_ids if prev_pos == 0 else None,
                text_attention_mask=text_attention_mask if prev_pos == 0 else None,
                past_key_values=kv_cache,
                input_pos=prev_pos,
            )
            logits = logits[:, -1]

            next_token = topk_sampling(
                logits, top_k=top_k, top_p=top_p, temperature=temperature
            )

            next_token = next_token.reshape(-1)
            next_token[eos_reached] = pad_id

            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_audio_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )

            tokens[:, cur_pos] = next_token

            stops = (torch.argmax(logits, dim=-1) == stop_id) | (next_token == stop_id)
            eos_reached |= (~input_audio_mask[:, cur_pos]) & stops
            eos_idx += ~eos_reached

            prev_pos = cur_pos
            if all(eos_reached):
                break
        out_tokens, out_lens = [], []
        for i, toks in enumerate(tokens):
            # cut to max gen len
            start = len(input_ids[i])
            toks = toks[start : eos_idx[i]]

            # cut to after eos tok if any

            # try:
            #     eos_idx = torch.nonzero(toks == stop_id)[0][0]
            #     toks = toks[:eos_idx]
            # except (ValueError, IndexError):
            #     pass
            out_tokens.append(toks)
            out_lens.append(len(toks))
        outs = pad_sequence(out_tokens, batch_first=True, padding_value=pad_id)
        outs_attention_mask = make_non_pad_mask(torch.tensor(out_lens)).to(
            dtype=attention_mask.dtype, device=device
        )
        if src_code_3d:
            outs = outs.unsqueeze(-1)
        return outs, outs_attention_mask


class NarDecoder(Decoder):
    def __init__(
        self,
        config: ValleConfig,
    ):
        super().__init__(config)
        self.num_codebooks = config.num_codebooks
        # self.train_qnt_rng = np.random.default_rng(42)
        self.lm_heads = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.codebook_size + 1, bias=False)
                for _ in range(config.num_codebooks - 1)
            ]
        )

        self.nar_stage_embeddings = nn.ModuleList(
            [
                TokenEmbedding(config.hidden_size, 1)
                for _ in range(self.num_codebooks - 1)
            ]
        )
        # starter: 0 -> random, exter: provider prompt
        self.prefix_mode = config.prefix_mode

        # We share the parameters of the output projection layer with the parameters of the acoustic embedding Wa
        # NOTE(Feiteng): In the experiment, this undermines accuracy
        # self.ar_predict_layer.weight = self.ar_audio_embedding.weight

        # We also share the parameters of the acoustic embedding layer and the output prediction layer,
        # which means the weights of the j-th prediction layer are the same as the (j + 1)-th acoustic embedding layer.
        # for j in range(0, self.num_codebooks - 2):
        #     self.lm_heads[j].weight = self.audio_embeddings[j + 2].weight
        for i in range(self.num_codebooks - 1):
            self.lm_heads[i].weight = self.audio_embeddings[i + 1].weight
        self.post_init()
        # apply special scaled init to the residual projections, GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("out_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers)
                )

    @property
    def is_ar(self) -> bool:
        return False

    def _preprocess_train_inputs(
        self,
        codebook_idx: int,
        input_ids: torch.Tensor,
        text_input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        prefix_codes: Optional[torch.Tensor] = None,
        prefix_attention_mask: Optional[torch.Tensor] = None,
    ):
        # 5.1 For the NAR acoustic prompt tokens, we select a random segment waveform of 3 seconds
        # from the same utterance.
        # We implement this differently.
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape[:2], device=device)
        if text_attention_mask is None:
            text_attention_mask = torch.ones(text_input_ids.shape[:2], device=device)
        text_embeds = self.text_embeddings(text_input_ids)

        text_len = text_input_ids.shape[1]
        text_position_ids = torch.arange(0, text_len, device=device).unsqueeze(0)
        text_embeds, text_inputs_pe = self.text_position(text_embeds, text_position_ids)

        if self.prefix_mode == "starter":
            # prefix at begining
            input_ids_len = attention_mask.long().sum(-1)
            int_low = (0.25 * input_ids_len.min()).type(torch.int64).item()
            prefix_len = torch.randint(int_low, int_low * 2, size=()).item()
            prefix_len = min(
                prefix_len, 3 * self.config.frame_rate
            )  # 24000/320 * 3s = 225 frames

            input_prompts = [
                input_embeds_layer(input_ids[:, :prefix_len, i]).unsqueeze(-1)
                for i, input_embeds_layer in enumerate(self.audio_embeddings)
            ]  # token embeddings of shape (b, t, n_embd)
            input_prompts = torch.cat(input_prompts, dim=-1)
            input_prompts = input_prompts.sum(dim=-1)

            input_embeds = [
                input_embeds_layer(input_ids[:, prefix_len:, i]).unsqueeze(-1)
                for i, input_embeds_layer in enumerate(
                    self.audio_embeddings[:codebook_idx]
                )
            ]  # token embeddings of shape (b, t, n_embd)

            # the input_embeddings are the sum of the j previous codebooks embeddings before
            # the current codebook_idx codebook
            input_embeds = torch.cat(input_embeds, dim=-1)
            input_embeds = input_embeds.sum(dim=-1)

            input_embeds = torch.cat([input_prompts, input_embeds], dim=1)
            position_ids = torch.arange(
                0, input_embeds.shape[1], device=device
            ).unsqueeze(0)
            input_embeds, input_embeds_pe = self.audio_position(
                input_embeds, position_ids
            )

            input_embeds = torch.cat([text_embeds, input_embeds], dim=1)
            input_embeds_pe = torch.cat([text_inputs_pe, input_embeds_pe], dim=1)
            attention_mask = torch.cat([text_attention_mask, attention_mask], dim=1)
            prefix_len += text_len

        else:
            assert (
                prefix_codes is not None
            ), f"prefix_mode={self.prefix_mode} requires exter codebook prompts."
            prefix_len = prefix_codes.shape[1]

            input_prompts = [
                input_embeds_layer(prefix_codes[..., i]).unsqueeze(-1)
                for i, input_embeds_layer in enumerate(self.audio_embeddings)
            ]  # token embeddings of shape (b, t, n_embd)
            input_prompts = torch.cat(input_prompts, dim=-1)
            input_prompts = input_prompts.sum(dim=-1)
            prefix_position_ids = torch.arange(
                0, input_prompts.shape[1], device=device
            ).unsqueeze(0)
            input_prompts, input_prompts_pe = self.audio_position(
                input_prompts, prefix_position_ids
            )

            input_embeds = [
                input_embeds_layer(input_ids[..., i]).unsqueeze(-1)
                for i, input_embeds_layer in enumerate(
                    self.audio_embeddings[:codebook_idx]
                )
            ]  # token embeddings of shape (b, t, n_embd)

            # the input_embeddings are the sum of the j previous codebooks embeddings before
            # the current codebook_idx codebook
            input_embeds = torch.cat(input_embeds, dim=-1)
            input_embeds = input_embeds.sum(dim=-1)
            position_ids = torch.arange(
                0, input_embeds.shape[1], device=device
            ).unsqueeze(0)
            input_embeds, input_embeds_pe = self.audio_position(
                input_embeds, position_ids
            )

            input_embeds_pe = torch.cat(
                [text_inputs_pe, input_prompts_pe, input_embeds_pe], dim=1
            )
            input_embeds = torch.cat([text_embeds, input_prompts, input_embeds], dim=1)
            if prefix_attention_mask is None:
                prefix_attention_mask = torch.ones(
                    input_prompts.shape[:2], device=device
                )

            attention_mask = torch.cat(
                [text_attention_mask, prefix_attention_mask, attention_mask], dim=1
            )
            prefix_len += text_len
        return input_embeds, attention_mask, input_embeds_pe, prefix_len

    def _preprocess_first_infer_inputs(
        self,
        input_ids: torch.Tensor,
        text_input_ids: Optional[torch.Tensor],
        prefix_codes: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        prefix_attention_mask: Optional[torch.Tensor] = None,
    ):
        assert input_ids.ndim == 3, input_ids.shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape[:2], device=device)
        if text_attention_mask is None:
            text_attention_mask = torch.ones(text_input_ids.shape[:2], device=device)

        text_embeds = self.text_embeddings(text_input_ids)

        text_len = text_input_ids.shape[1]
        text_position_ids = torch.arange(0, text_len, device=device).unsqueeze(0)
        input_prompts = [
            input_embeds_layer(prefix_codes[..., i]).unsqueeze(-1)
            for i, input_embeds_layer in enumerate(self.audio_embeddings)
        ]  # token embeddings of shape (b, t, n_embd)
        input_prompts = torch.cat(input_prompts, dim=-1)
        input_prompts = input_prompts.sum(dim=-1)
        prefix_position_ids = torch.arange(
            0, input_prompts.shape[1], device=device
        ).unsqueeze(0)
        # token embeddings of shape (b, t, n_embd)
        input_embeds = self.audio_embeddings[0](input_ids[..., 0])

        if self.prefix_mode == "starter":
            prompts_lens = (
                prefix_codes.shape[1]
                if prefix_attention_mask is None
                else prefix_attention_mask.sum(dim=-1, keepdim=True)
            )
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids += prompts_lens
            # align first(batch) dim
            text_position_ids = text_position_ids.expand(
                position_ids.shape[0], text_position_ids.shape[1]
            )
            prefix_position_ids = prefix_position_ids.expand(
                position_ids.shape[0], prefix_position_ids.shape[1]
            )
        else:
            position_ids = torch.arange(
                0, input_embeds.shape[1], device=device
            ).unsqueeze(0)

        input_embeds = torch.cat([text_embeds, input_prompts, input_embeds], dim=1)
        if prefix_attention_mask is None:
            prefix_attention_mask = torch.ones(input_prompts.shape[:2], device=device)

        attention_mask = torch.cat(
            [text_attention_mask, prefix_attention_mask, attention_mask], dim=1
        )
        position_ids = torch.cat(
            [text_position_ids, prefix_position_ids, position_ids], dim=1
        )

        return input_embeds, attention_mask, position_ids

    def forward(
        self,
        codebook_idx: int,  # an additionnal idx corresponding to the id of the codebook that will be predicted
        input_ids: Optional[torch.Tensor],
        text_input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        prefix_codes: Optional[torch.Tensor] = None,
        prefix_attention_mask: Optional[torch.Tensor] = None,
    ):
        """Forward func for training.

        Args:
            codebook_idx (`int`):
                Index of the codebook that will be predicted.
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length, number_of_codebooks)`):
                Indices of input sequence tokens of audio code embeddings.

            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding audio token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

            text_input_ids (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`):
                Indices of input sequence text tokens in the vocabulary.  should
                you provide it.
            text_attention_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.
            prefix_codes (`torch.LongTensor` of shape `(batch_size, prefix_codes_sequence_length)` *optional*):
                Indices of externel audio prompts tokens.
            prefix_attention_mask (`torch.Tensor` of shape `(batch_size, prefix_codes_sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.
        """

        if codebook_idx == 0:
            raise ValueError(
                "Cannot predict 0th codebook - 0th codebook should be predicted by the AR/Encodec model"
            )

        assert input_ids.shape[-1] == self.num_codebooks, input_ids.shape
        # codebook_idx = int(self.train_qnt_rng.integers(1, self.num_codebooks))

        (
            input_embeds,
            attention_mask,
            input_embeds_pe,
            prefix_len,
        ) = self._preprocess_train_inputs(
            codebook_idx,
            input_ids,
            text_input_ids=text_input_ids,
            attention_mask=attention_mask,
            text_attention_mask=text_attention_mask,
            prefix_codes=prefix_codes,
            prefix_attention_mask=prefix_attention_mask,
        )

        if not self.rope:
            input_embeds_pe = None
        stage_embedding = self.nar_stage_embeddings[codebook_idx - 1].weight
        attention_4d_mask = prepare_4d_attention_mask(
            attention_mask, input_embeds.dtype
        )
        attention_4d_mask = _cast_attn_bias(attention_4d_mask, input_embeds.dtype)
        hidden_states = input_embeds.type_as(attention_4d_mask)
        for decoder_layer in self.layers:

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_4d_mask,
                rotary_pe=input_embeds_pe,
                stage_embedding=stage_embedding,
            )

            hidden_states = layer_outputs

        hidden_states = self.norm_f(hidden_states, stage_embedding)

        # (B, T, codebook_size + 1)
        logits = self.lm_heads[codebook_idx - 1](hidden_states[:, prefix_len:])

        return logits

    def generate(
        self,
        input_ids: Optional[torch.Tensor],
        text_input_ids: Optional[torch.Tensor],
        prefix_codes: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        prefix_attention_mask: Optional[torch.Tensor] = None,
    ):
        """Greedy search NAR model.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length, 1)`):
                The first tokens of audio codebooks.

            text_input_ids (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`):
                Indices of input sequence text tokens in the vocabulary.
            prefix_codes (`torch.LongTensor` of shape `(batch_size, prefix_codes_sequence_length)` *optional*):
                Indices of externel audio prompts tokens.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding audio token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

            text_attention_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.
            prefix_attention_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.
        """
        if input_ids.ndim == 2:
            input_ids = input_ids.unsqueeze(-1)
        assert input_ids.shape[-1] == 1, input_ids.shape
        text_len, pred_len = text_input_ids.shape[1], input_ids.shape[1]

        (
            input_embeds,
            attention_mask,
            position_ids,
        ) = self._preprocess_first_infer_inputs(
            input_ids,
            text_input_ids,
            prefix_codes=prefix_codes,
            attention_mask=attention_mask,
            text_attention_mask=text_attention_mask,
            prefix_attention_mask=prefix_attention_mask,
        )
        # buff preds embed part without positional
        pred_buff = input_embeds[:, -pred_len:]

        input_embeds[:, :text_len], text_embeds_pe = self.text_position(
            input_embeds[:, :text_len], position_ids[:, :text_len]
        )
        input_embeds[:, text_len:], audio_embeds_pe = self.audio_position(
            input_embeds[:, text_len:], position_ids[:, text_len:]
        )

        input_embeds_pe = (
            torch.cat([text_embeds_pe, audio_embeds_pe], dim=1) if self.rope else None
        )

        attention_4d_mask = prepare_4d_attention_mask(
            attention_mask, input_embeds.dtype
        )
        preds = F.pad(
            input_ids,
            (0, self.num_codebooks - 1),
            "constant",
            self.config.codebook_size,
        )

        for codebook_idx in range(1, self.num_codebooks):
            stage_embedding = self.nar_stage_embeddings[codebook_idx - 1].weight
            hidden_states = input_embeds
            for decoder_layer in self.layers:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_4d_mask,
                    rotary_pe=input_embeds_pe,
                    stage_embedding=stage_embedding,
                )

                hidden_states = layer_outputs

            hidden_states = self.norm_f(hidden_states, stage_embedding)

            # (B, T, codebook_size + 1)
            logits = self.lm_heads[codebook_idx - 1](hidden_states[:, -pred_len:])
            relevant_logits = logits[..., : self.config.codebook_size]
            codebook_preds = torch.argmax(relevant_logits, -1)
            codebook_preds, _ = padding_codes(
                codebook_preds, attention_mask[:, -pred_len:], self.config.codebook_size
            )
            preds[..., codebook_idx] = codebook_preds

            # prepare embed for next iter (1 -> 6).
            if codebook_idx != self.num_codebooks - 1:
                pred_buff += self.audio_embeddings[codebook_idx](codebook_preds)
                input_embeds[:, -pred_len:], _ = self.audio_position(
                    pred_buff, position_ids[:, -pred_len:]
                )

        return preds


# https://github.com/lifeiteng/vall-e/blob/main/valle/models/valle.py
def top_k_top_p_filtering(
    logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits


def topk_sampling(logits, top_k=10, top_p=1.0, temperature=1.0):
    """
    Args:
        logits (torch.Tensor):
            logits distribution.
        topk (int, optional):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
            if negative, ignore topk filter. Default to -100.
        temperature (float, optional):
            Temperature value for controlling randomness in sampling. Defaults to 1.0.
        top_p (float, optional):
            Top-p probability threshold for nucleus sampling. Defaults to 1.
    """
    # Temperature (higher temperature => more likely to sample low probability tokens)
    if temperature != 1.0:
        logits = logits / temperature
    # Top-p/top-k filtering

    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    # Sample
    token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    return token


def padding_codes(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    pad_value: Any,
    shift_tgt_eos: Optional[Any] = None,
):
    """Padding codebooks with padding id.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length, number_of_codebooks)`):
            Indices of input sequence tokens of audio code embeddings.

        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`):
            Mask to avoid performing attention on padding audio token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        pad_value:
            Padding mask value.
        shift_tgt_eos (*optional*):
            If given not None, shift left and pad eos as casual target.
    """
    src_2d = len(input_ids.shape) == 2
    if src_2d:
        input_ids = input_ids.unsqueeze(-1)
    if attention_mask is not None:
        if tuple(attention_mask.shape) != tuple(input_ids.shape[:2]):
            raise ValueError(
                f"Incorrect 2D attention_mask shape: {tuple(attention_mask.shape)}; expected: {input_ids.shape[:2]}."
            )

        # (B, T, 1)
        inversed_attention_mask = (~(attention_mask.bool())).unsqueeze(-1)
        input_ids = input_ids.masked_fill(inversed_attention_mask, pad_value)

    tgt_ids = None
    if shift_tgt_eos is not None:
        tgt_ids = input_ids.new_zeros(input_ids.shape)
        tgt_ids[:, :-1] = input_ids[:, 1:].clone()
        tgt_ids[:, -1] = pad_value
        if pad_value != shift_tgt_eos:
            lens = (
                input_ids.shape[1]
                if attention_mask is None
                else attention_mask.sum(-1).long()
            )
            tgt_ids[[i for i in range(tgt_ids.shape[0])], lens - 1] = shift_tgt_eos
        tgt_ids = tgt_ids.squeeze(-1) if src_2d else tgt_ids
        # input_ids = input_ids[[i for i in range(input_ids.shape[0])],lens-1]
        # input_ids = input_ids.transpose(1, 2)
        # input_ids = F.pad(input_ids, (0, 1), value=pad_value)
        # input_ids = input_ids.transpose(1, 2)
        # input_ids.squeeze(-1) if src_2d else input_ids
        # # inputs, targets
        # return input_ids[:, :-1], input_ids[:, 1:]

    return input_ids.squeeze(-1) if src_2d else input_ids, tgt_ids


def _cast_attn_bias(bias: torch.Tensor, input_dtype: torch.dtype) -> torch.Tensor:
    target_dtype = input_dtype
    if torch.is_autocast_enabled():
        if bias.device.type == "cuda":
            target_dtype = torch.get_autocast_gpu_dtype()
        elif bias.device.type == "cpu":
            target_dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
    if bias.dtype != target_dtype:
        bias = bias.to(target_dtype)
        bias.masked_fill_(bias == float("-inf"), torch.finfo(target_dtype).min)
    return bias
