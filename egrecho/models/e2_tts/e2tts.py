# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2024-08)
"""
E2TTS implementation.

Refs:
    paper: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS
       https://arxiv.org/abs/2406.18009
    repo: https://github.com/lucidrains/e2-tts-pytorch/tree/main
"""
import contextlib
import math
from copy import deepcopy
from functools import partial
from random import random
from typing import Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from egrecho.core.model_base import ModelBase
from egrecho.models.e2_tts.e2tts_config import DurPredConfig, E2TTSConfig, ModelConfig
from egrecho.nn.activation import Nonlinearity
from egrecho.utils.common import alt_none
from egrecho.utils.cuda_utils import avoid_float16_autocast_context
from egrecho.utils.imports import is_package_available
from egrecho.utils.mask import make_non_pad_mask, prepare_4d_attention_mask

if not is_package_available('torchdiffeq'):
    raise ImportError(f'Require torchdiffeq, try pip install')
from torchdiffeq import odeint

MaskCache = torch.Tensor
PECache = torch.Tensor
KVCache = Tuple[torch.Tensor, torch.Tensor]


#################################################################################
#                            Embedding Layers                                   #
#################################################################################


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
        scale: bool = False,
        alpha: bool = False,
        device=None,
    ):
        super().__init__()
        assert d_model % 2 == 0
        self.d_model = d_model
        self.x_scale = math.sqrt(d_model) if scale else 1.0
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=alpha)
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
        return output, pos_emb


def apply_rope(x: torch.Tensor, rotary_pe: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x (`torch.Tensor`): The tensor of shape [bsz, n_head, T, d_model//n_head].
        rotary_emed (`torch.Tensor`): Precomputed rotary of shape [b_rope, T, d_model//n_head//2, 2].
    """

    assert x.shape[2] == rotary_pe.shape[1]

    xshaped = x.reshape(*x.shape[:-1], -1, 2)  # (bsz, n_head, T, d_model//n_head//2, 2)
    rotary_pe = rotary_pe.unsqueeze(1)  # b_rope, 1, T, d_model//n_head//2, 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rotary_pe[..., 0] - xshaped[..., 1] * rotary_pe[..., 1],
            xshaped[..., 1] * rotary_pe[..., 0] + xshaped[..., 0] * rotary_pe[..., 1],
        ],
        -1,
    )
    return x_out2.flatten(3).type_as(x)


class RotaryEmbedding(nn.Module):
    """Rotary positional encoding module.
    RoFormer: Enhanced Transformer with Rotary Position Embedding.

    Args:
        d_head (int): per qkv head dimension.
    """

    def __init__(
        self,
        d_head: int,
        device=None,
        use_xpos=False,
        ntk_max_position: int = 8192,
        scaling_factor=1.0,
        base: float = 10000.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.d_head = d_head
        self.base = base
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.d_head, 2, dtype=torch.int64).float().to(device)
                / self.d_head
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.ntk_max_position = ntk_max_position
        self.use_xpos = use_xpos

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

    # copied from transformers.models.llama.LlamaDynamicNTKScalingRotaryEmbedding
    def _forward_ntk(self, x: torch.Tensor, position_ids: torch.Tensor):
        """RotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""
        # difference to the original RoPE: inv_freq is recomputed when the sequence length > original length
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.ntk_max_position:
            base = self.base * (
                (self.scaling_factor * seq_len / self.ntk_max_position)
                - (self.scaling_factor - 1)
            ) ** (self.d_head / (self.d_head - 2))
            inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(0, self.d_head, 2, dtype=torch.int64)
                    .float()
                    .to(x.device)
                    / self.d_head
                )
            )
            self.register_buffer(
                "inv_freq", inv_freq, persistent=False
            )  # TODO joao: this may break with compilation

        cos, sin = self._forward(x, position_ids)
        return cos, sin

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
        if not self.use_xpos:
            cos, sin = self._forward(x, position_ids)
        else:
            cos, sin = self._forward_ntk(x, position_ids)
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


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size):
        super().__init__()
        assert hidden_size % 2 == 0, hidden_size
        half_dim = hidden_size // 2
        self.weights = nn.Parameter(torch.randn(half_dim))
        self.mlp = nn.Sequential(
            nn.Linear(1 + hidden_size, hidden_size, bias=True),
            nn.SiLU(),
        )

    def forward(self, t: Tensor):
        if t.ndim == 1:
            t = t[..., None]

        assert t.ndim == 2, t.ndim
        freqs = t * self.weights * 2 * math.pi
        fouriered = torch.cat((t, freqs.sin(), freqs.cos()), dim=-1)
        return self.mlp(fouriered)


# Tok embedding
class TextEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size,
        vocab_size=256,
        padding_id: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = TokenEmbedding(hidden_size, vocab_size)
        self.padding_id = padding_id

    def forward(
        self,
        text: torch.LongTensor,
        max_seq_len: int,
    ) -> Tensor:
        text = text[
            :, :max_seq_len
        ]  # just curtail if character tokens are more than the mel spec tokens, one of the edge cases the paper did not address
        text = F.pad(text, (0, max_seq_len - text.shape[1]), value=self.padding_id)
        return self.embed(text)


class TextAudioCrossCondition(nn.Module):
    def __init__(self, dim, dim_text, cond_audio_to_text=True):
        super().__init__()
        self.text_to_audio = nn.Linear(dim_text + dim, dim, bias=False)
        nn.init.zeros_(self.text_to_audio.weight)

        self.cond_audio_to_text = cond_audio_to_text

        if cond_audio_to_text:
            self.audio_to_text = nn.Linear(dim + dim_text, dim_text, bias=False)
            nn.init.zeros_(self.audio_to_text.weight)
        self._init_wt()

    def _init_wt(self):
        nn.init.zeros_(self.text_to_audio.weight)
        if self.cond_audio_to_text:
            nn.init.zeros_(self.audio_to_text.weight)

    def forward(self, audio: Tensor, text: Tensor):
        """cross condition

        Args:
            audio:
                audio hidden states (B, T, D)
            text:
                text hidden states (B, T, D_TEXT)
        """
        audio_text = torch.cat((audio, text), dim=-1)

        text_cond = self.text_to_audio(audio_text)
        audio_cond = self.audio_to_text(audio_text) if self.cond_audio_to_text else 0.0

        return audio + text_cond, text + audio_cond


# copied from https://github.com/bfs18/e2_tts/blob/main/rfwave/dit.py
class Base2FourierFeatures(nn.Module):
    def __init__(self, start=0, stop=8, step=1):
        super().__init__()
        self.start = start
        self.stop = stop
        self.step = step

    def __call__(self, inputs):
        inputs = inputs.transpose(1, 2)
        freqs = range(self.start, self.stop, self.step)

        # Create Base 2 Fourier features
        w = (
            2.0 ** (torch.tensor(freqs, dtype=inputs.dtype)).to(inputs.device)
            * 2
            * torch.pi
        )
        w = torch.tile(w[None, :, None], (1, inputs.shape[1], 1))

        # Compute features
        h = torch.repeat_interleave(inputs, len(freqs), dim=1)
        h = w * h
        h = torch.stack([torch.sin(h), torch.cos(h)], dim=2)

        h = h.reshape(h.size(0), -1, h.size(3))

        return h.transpose(1, 2)


############################################################################
#                            Norm Layers                                   #
############################################################################
def norm_fac(
    name: Literal["rms", "adaln_act", "adaln", "adaln0", "adaln0_pre"],
    config: ModelConfig,
):
    if name.lower() == "rms":
        return RMSNorm(config.hidden_size, eps=config.norm_eps)
    elif name == "adaln":
        return AdaLN(config.hidden_size, eps=config.norm_eps)
    elif name == "adaln_act":
        return AdaLN(config.hidden_size, act_cond=True, eps=config.norm_eps)
    elif name == "adaln0":
        return AdaLNZero(
            config.hidden_size,
        )
    elif name == "adaln0_pre":
        return AdaLNZero(config.hidden_size, zero_post=False)
    else:
        raise ValueError(
            f"Invalid norm name={name}, choose from rms|adaln|adaln0|adaln0_pre|adaln_act "
        )


class AdaLNZero(nn.Module):
    """adaln zero part from DiT paper"""

    def __init__(self, dim, dim_condition=None, zero_post: bool = True):
        super().__init__()
        dim_condition = dim_condition or dim
        self.to_gamma = nn.Linear(dim_condition, dim)
        self.zero_post = zero_post
        self._init_wt()

    def _init_wt(self):
        nn.init.zeros_(self.to_gamma.weight)
        bias_init_value = 0.0 if self.zero_post else 1.0
        nn.init.constant_(self.to_gamma.bias, bias_init_value)

    def forward(self, x, condition):
        if condition.ndim == 2:
            condition = condition.unsqueeze(1)

        gamma = self.to_gamma(condition)
        return x * gamma


class AdaLN(nn.Module):
    """adaln part from DiT paper"""

    def __init__(
        self, dim, dim_condition=None, act_cond: bool = False, eps: float = 1e-6
    ):
        super().__init__()
        dim_condition = dim_condition or dim
        self.ln = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.cond_linear = nn.Linear(dim_condition, dim * 2)
        self.act_cond = act_cond
        self._init_wt()

    def _init_wt(self):
        nn.init.zeros_(self.cond_linear.weight)
        nn.init.zeros_(self.cond_linear.bias)

    def forward(self, x, condition, act_cond: bool = False):
        xdtype = x.dtype
        if condition.ndim == 2:
            condition = condition.unsqueeze(1)
        if act_cond:
            condition = F.silu(condition)
        gamma, beta = self.cond_linear(condition).chunk(2, dim=-1)
        x = self.ln(x)
        return (x * (1.0 + gamma) + beta).to(xdtype)


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(d_model))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x, condition: Tensor = torch.empty(0)):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


############################################################################
#                              Components                                  #
############################################################################


def softclamp(t, value):
    return (t / value).tanh() * value


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Attention(nn.Module):

    sdpa_32bit = False

    def __init__(self, config: ModelConfig):
        super().__init__()

        # regularization
        self.dropout = config.attn_dropout
        self.attn_dropout = nn.Dropout(self.dropout)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.d_head
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_heads must be divisible by num_key_value_heads (got `num_heads`: {self.num_heads}"
                f" and `num_key_value_heads`: {self.num_key_value_heads})."
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

        self.qk_norm = config.qk_norm
        self.q_norm = RMSNorm(self.head_dim) if config.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if config.qk_norm else nn.Identity()
        # upcast attention to fp32
        self.attn_fn = (
            partial(F.softmax, dtype=torch.float32) if not config.qk_norm else F.softmax
        )
        self.softclamp_logits = config.softclamp_logits and config.softclamp_logits_val
        if self.softclamp_logits:
            assert (
                not self.spda_att
            ), "flash sdpa attention not compatible with logit softclamp value yet. set softclamp_logits to None or close sdpa attn"
            assert config.softclamp_logits_val > 0.0, config.softclamp_logits_val
            self.softclamp_logits_val = config.softclamp_logits_val

        # key, query, value projections
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.qkv_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.qkv_bias,
        )

        # output projection
        self.out_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
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

        if self.softclamp_logits:
            attn_weights = softclamp(attn_weights, self.softclamp_logits_val)

        if attention_mask is not None:
            # Apply the attention mask
            attention_mask = _cast_attn_bias(attention_mask, attn_weights.dtype)
            attn_weights = attn_weights + attention_mask

        # attn_weights = torch.max(
        #     attn_weights,
        #     torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device),
        # )

        attn_weights = self.attn_fn(attn_weights, dim=-1).to(query.dtype)

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
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # (bsz, head, seq, dim)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_key_value_heads, self.head_dim)
        value = self._split_heads(value, self.num_key_value_heads, self.head_dim)
        if self.qk_norm:
            query, key = self.q_norm(query), self.k_norm(key)
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

        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)

        attn_output = self._attn(query, key, value, attention_mask=attention_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)

        outputs = (attn_output, present)

        return outputs


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
        activation_type="gelu_tanh",
        bias=False,
        multiple_of: int = 256,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.dropout = nn.Dropout(dropout_rate)
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        if activation_type == "gelu_tanh":
            self.activation = nn.GELU(approximate="tanh")
        else:
            self.activation = Nonlinearity(activation_type)

    def forward(self, x):
        return self.out_proj(self.dropout(self.activation(self.w1(x)) * self.w3(x)))


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, config: E2TTSConfig, unet_in: bool = False):
        super().__init__()

        self.is_unet_right = config.unet_mode and unet_in
        if self.is_unet_right:
            self.unet_proj = nn.Linear(
                config.hidden_size * 2, config.hidden_size, bias=False
            )

        self.norm1 = norm_fac("adaln", config)
        self.attn = Attention(config)
        self.post_att_norm = (
            norm_fac("rms", config) if config.sandwish_norm else nn.Identity()
        )
        self.adaln_zero1 = norm_fac("adaln0_pre", config)
        self.norm2 = norm_fac("adaln", config)
        self.mlp = GatedFFN(
            config.hidden_size,
            4 * config.hidden_size,
            dropout_rate=config.dropout,
        )
        self.post_mlp_norm = (
            norm_fac("rms", config) if config.sandwish_norm else nn.Identity()
        )

        self.adaln_zero2 = norm_fac("adaln0", config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cond_embedding: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pe: Optional[torch.Tensor] = None,
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
        cond_embedding = F.silu(cond_embedding)
        if self.is_unet_right:
            hidden_states = self.unet_proj(hidden_states)

        attn_output, _ = self.attn(
            self.norm1(hidden_states, cond_embedding),
            attention_mask=attention_mask,
            rotary_pe=rotary_pe,
        )
        attn_output = self.post_att_norm(attn_output)
        attn_output = self.adaln_zero1(attn_output, cond_embedding)

        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.adaln_zero2(
            self.post_mlp_norm(
                self.mlp(self.norm2(hidden_states, cond_embedding)),
            ),
            cond_embedding,
        )
        return hidden_states


class TransformerBlock(nn.Module):
    """
    A Transformer block without condition.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = norm_fac("rms", config)
        self.attn = Attention(config)
        self.post_att_norm = (
            norm_fac("rms", config) if config.sandwish_norm else nn.Identity()
        )
        self.norm2 = norm_fac("rms", config)
        self.mlp = GatedFFN(
            config.hidden_size,
            4 * config.hidden_size,
            dropout_rate=config.dropout,
        )
        self.post_mlp_norm = (
            norm_fac("rms", config) if config.sandwish_norm else nn.Identity()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pe: Optional[torch.Tensor] = None,
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
            self.norm1(hidden_states),
            attention_mask=attention_mask,
            rotary_pe=rotary_pe,
        )
        attn_output = self.post_att_norm(attn_output)

        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.post_mlp_norm(
            self.mlp(self.norm2(hidden_states)),
        )

        return hidden_states


class CondTextBlock(TransformerBlock):
    def __init__(
        self, config: ModelConfig, d_audio: int, cond_audio_to_text: bool = True
    ):
        super().__init__(config)
        self.cross_condition = TextAudioCrossCondition(
            dim=d_audio,
            dim_text=config.hidden_size,
            cond_audio_to_text=cond_audio_to_text,
        )

    def forward(
        self,
        audio_hidden_states: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pe: Optional[torch.Tensor] = None,
        cond_on: bool = True,
        **kwargs,
    ):
        """Cross condition of audio and text.

        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, 1, query_sequence_length, key_sequence_length)`.
            rotary_pe (`torch.Tensor`):
                If not None, apply precomputed rotary of shape [bsz, T, d_model//n_head//2, 2].
        """
        h_text = super().forward(hidden_states, attention_mask, rotary_pe, **kwargs)
        h_audio, h_text = self.cross_condition(audio_hidden_states, h_text)

        # TRACE available
        cond_on = int(bool(cond_on))
        h_audio = cond_on * h_audio + (1 - cond_on) * audio_hidden_states
        h_text = cond_on * h_text + (1 - cond_on) * hidden_states
        return h_audio, h_text


############################################################################
#                               BackBone                                   #
############################################################################


class DiTBackBone(nn.Module):
    """Dit backbone"""

    def __init__(self, config: E2TTSConfig):
        super().__init__()

        if config.unet_mode:
            if config.num_layers % 2 != 0:
                raise ValueError(
                    f"Unet num_layers needs to be even, but got {config.num_layers}"
                )
        self.unet_mode = config.unet_mode

        self.with_fourier_features = config.with_fourier_features
        self.cond_inp_add = config.cond_inp_add
        self.cond_text_net = config.cond_text_net
        self.num_layers = config.num_layers

        self.proj_in = nn.Linear(config.inputs_dim, config.hidden_size)
        self.cond_proj_in = nn.Linear(config.inputs_dim, config.hidden_size)
        if self.with_fourier_features:
            fourier_module = Base2FourierFeatures(start=6, stop=8)
            fourier_dim = (
                config.inputs_dim
                * 2
                * ((fourier_module.stop - fourier_module.start) // fourier_module.step)
            )
            self.fourier_tfm = nn.Sequential(
                fourier_module,
                nn.Linear(fourier_dim, config.hidden_size),
            )

        num_concat = (
            1
            + int(not config.cond_inp_add)
            + int(config.with_fourier_features)
            + int(not self.cond_text_net)
        )

        self.to_embed = (
            nn.Linear(
                config.hidden_size * num_concat,
                config.hidden_size,
            )
            if num_concat > 1
            else nn.Identity()
        )
        self.norm_embed = norm_fac("adaln_act", config)

        self.rotary_embedder = RotaryEmbedding(
            config.d_head,
        )
        self.layers = nn.ModuleList(
            [
                DiTBlock(
                    config,
                    unet_in=(self.unet_mode and layer_idx >= config.num_layers // 2),
                )
                for layer_idx in range(config.num_layers)
            ]
        )
        if self.cond_text_net:
            self.text_rotary_embedder = RotaryEmbedding(
                config.d_head,
            )
            text_config = deepcopy(config)
            text_config.backbone_sizes.hidden_size = (
                text_config.backbone_sizes.hidden_size // 2
            )
            self.cond_text_layers = nn.ModuleList(
                [
                    CondTextBlock(
                        text_config,
                        d_audio=config.hidden_size,
                        cond_audio_to_text=i < (config.num_layers - 1),
                    )
                    for i in range(config.num_layers)
                ]
            )

        self.norm_f = norm_fac("rms", config)

    def forward(
        self,
        x_t,
        t_emb,
        x_ctx,
        z_emb,
        attention_mask: Optional[torch.Tensor] = None,
        cond_on_text: bool = True,
    ):
        """Forward backbone.

        Args:
            x_t:
                sample at flow step t (B, T, F)
            t_emb:
                flow step t embed. (B, D)
            x_ctx:
                condition masked sample. (B, T, F)
            z_emb:
                condition text embed (B, T, D)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding features. Mask values selected in `[0, 1]`:

                - 1 for **not masked**,
                - 0 for **masked**.
            cond_on_text:
                force control ignore z_emb
        """

        z_emb = int(bool(cond_on_text)) * z_emb
        if self.with_fourier_features:
            x_t_f = self.fourier_tfm(x_t)
        else:
            x_t_f = None
        x_t = self.proj_in(x_t)
        x_ctx = self.cond_proj_in(x_ctx)
        if self.cond_inp_add:
            x_t = x_t + x_ctx
            to_concat = [x_t]
        else:
            to_concat = [x_t, x_ctx]
        if not self.cond_text_net:

            to_concat.append(z_emb)
        if x_t_f is not None:
            to_concat.append(x_t_f)

        x = self.to_embed(torch.cat(to_concat, dim=-1))

        position_ids = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        _, rotary_pe = self.rotary_embedder(x, position_ids)

        # TODO: left space to condition on more
        c = t_emb

        x = self.norm_embed(x, c)
        if torch.is_autocast_enabled():
            if x.device.type == "cuda":
                x = x.to(torch.get_autocast_gpu_dtype())
            elif x.device.type == "cpu":
                x = x.to(torch.get_autocast_cpu_dtype())
        if attention_mask is not None:
            attention_4d_mask = prepare_4d_attention_mask(attention_mask, x.dtype)
        else:
            attention_4d_mask = None

        hidden_states = x
        if self.cond_text_net and cond_on_text:
            z_position_ids = torch.arange(
                z_emb.shape[1], device=z_emb.device
            ).unsqueeze(0)
            _, rotary_txt_pe = self.text_rotary_embedder(z_emb, z_position_ids)
            text_hidden_states = z_emb.type_as(hidden_states)

        if self.unet_mode:
            skips = []

        for layer_i in range(self.num_layers):

            # cond text attn
            if self.cond_text_net and cond_on_text:
                cond_text_layer = self.cond_text_layers[layer_i]
                hidden_states, text_hidden_states = cond_text_layer(
                    hidden_states,
                    text_hidden_states,
                    rotary_pe=rotary_txt_pe,
                    attention_mask=attention_4d_mask,
                    cond_on=cond_on_text,
                )

            # unet skip
            if self.unet_mode:
                is_right_half = layer_i >= (self.num_layers // 2)
                is_left_half = not is_right_half

                if is_left_half:
                    skips.append(hidden_states)
                if is_right_half:
                    skip = skips.pop()
                    hidden_states = torch.cat((hidden_states, skip), dim=-1)

            # main attn
            layer = self.layers[layer_i]
            hidden_states = layer(
                hidden_states,
                cond_embedding=c,
                attention_mask=attention_4d_mask,
                rotary_pe=rotary_pe,
            )
        hidden_states = self.norm_f(hidden_states, c)
        return hidden_states


class DurationBackBone(nn.Module):
    """Transformer backbone"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.cond_text_net = config.cond_text_net
        self.num_layers = config.num_layers
        self.proj_in = nn.Linear(config.inputs_dim, config.hidden_size)
        num_concat = 1 + int(not self.cond_text_net)
        self.to_embed = (
            nn.Linear(
                config.hidden_size * num_concat,
                config.hidden_size,
            )
            if num_concat > 1
            else nn.Identity()
        )
        self.rotary_embedder = RotaryEmbedding(
            config.d_head,
        )
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        if self.cond_text_net:
            self.text_rotary_embedder = RotaryEmbedding(
                config.d_head,
            )
            text_config = deepcopy(config)
            text_config.backbone_sizes.hidden_size = (
                text_config.backbone_sizes.hidden_size // 2
            )
            self.cond_text_layers = nn.ModuleList(
                [
                    CondTextBlock(
                        text_config,
                        d_audio=config.hidden_size,
                        cond_audio_to_text=i < (config.num_layers - 1),
                    )
                    for i in range(config.num_layers)
                ]
            )

        self.norm_f = norm_fac("rms", config)

    def forward(
        self,
        x,
        text_emb,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Forward backbone.

        Args:
            x:
               feature inputs (B, T, F)
            text_emb:
                text embd (B, T, D)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding features. Mask values selected in `[0, 1]`:

                - 1 for **not masked**,
                - 0 for **masked**.

        """

        x = self.proj_in(x)
        to_concat = [x]
        if not self.cond_text_net:
            to_concat.append(text_emb)
        x = self.to_embed(torch.cat(to_concat, dim=-1))

        position_ids = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        _, rotary_pe = self.rotary_embedder(x, position_ids)

        if torch.is_autocast_enabled():
            if x.device.type == "cuda":
                x = x.to(torch.get_autocast_gpu_dtype())
            elif x.device.type == "cpu":
                x = x.to(torch.get_autocast_cpu_dtype())
        if attention_mask is not None:
            attention_4d_mask = prepare_4d_attention_mask(attention_mask, x.dtype)
        else:
            attention_4d_mask = None

        hidden_states = x
        if self.cond_text_net:
            txt_position_ids = torch.arange(
                text_emb, text_emb.shape[1], device=text_emb.device
            ).unsqueeze(0)
            _, rotary_txt_pe = self.text_rotary_embedder(text_emb, txt_position_ids)
            text_hidden_states = text_emb.type_as(hidden_states)
        for layer_i in range(self.num_layers):
            if self.cond_text_net:
                cond_text_layer = self.cond_text_layers[layer_i]
                hidden_states, text_hidden_states = cond_text_layer(
                    hidden_states,
                    text_hidden_states,
                    rotary_pe=rotary_txt_pe,
                    attention_mask=attention_4d_mask,
                )
            layer = self.layers[layer_i]
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_4d_mask,
                rotary_pe=rotary_pe,
            )
        hidden_states = self.norm_f(hidden_states)

        return hidden_states


############################################################################
#                                 Models                                   #
############################################################################


class E2TTSBase(ModelBase):
    """
    Abstract base module for valle.
    """

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (AdaLNZero, AdaLN, TextAudioCrossCondition)):
            module._init_wt()


class E2TTSFlow(E2TTSBase):
    def __init__(
        self,
        config: E2TTSConfig,
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.text_embedding = TextEmbedding(
            config.hidden_size // 2 if config.cond_text_net else config.hidden_size,
            config.vocab_size,
            config.pad_text_token_id,
        )
        self.null_cond = torch.zeros(
            config.hidden_size // 2 if config.cond_text_net else config.hidden_size
        )
        self.t_embedder = TimestepEmbedder(config.hidden_size)

        self.backbone = DiTBackBone(config=config)

        self.to_pred = nn.Linear(config.hidden_size, config.inputs_dim)

        self.post_init()
        # apply special scaled init to the residual projections, GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("out_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers)
                )

    def backbone_with_pred_head(
        self,
        x_t,
        x_ctx,
        times,
        mask: Optional[torch.Tensor] = None,
        text_input_ids: Optional[torch.Tensor] = None,
        drop_text_cond: Optional[bool] = None,
    ):
        """Forward backbone.

        Args:
            x_t:
                sample at flow step t (B, T, F)
            x_ctx:
                condition masked sample. (B, T, F)
            times:
                flow timestep (B,)
            mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding features. Mask values selected in `[0, 1]`:

                - 1 for **not masked**,
                - 0 for **masked**.
            text_input_ids (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`):
                Indices of input sequence text tokens in the vocabulary.  should
                you provide it.
            drop_text_cond:
                Whether drop condition text
        """
        b, seq_len = x_t.shape[:2]

        if times.ndim == 0:
            times = times.expand(b)
        t_emb = self.t_embedder(times)

        # whether to use a text embedding
        drop_text_cond = alt_none(
            drop_text_cond, self.training and random() < self.config.cond_dropout
        )
        cond_on_text = text_input_ids is not None and not drop_text_cond
        if cond_on_text:
            text_embed = self.text_embedding(text_input_ids, seq_len)
        else:
            text_embed = self.text_embedding(
                torch.full(
                    (b, seq_len), self.config.pad_text_token_id, device=x_t.device
                ).long(),
                seq_len,
            )

        attended = self.backbone(
            x_t,
            t_emb=t_emb,
            x_ctx=x_ctx,
            z_emb=text_embed,
            attention_mask=mask,
            cond_on_text=cond_on_text,
        )

        return self.to_pred(attended)

    def cfg_backbone_with_pred_head(
        self,
        *args,
        cfg_strength: float = 1.0,
        **kwargs,
    ):

        pred = self.backbone_with_pred_head(*args, drop_text_cond=False, **kwargs)

        if cfg_strength < 1e-5:
            return pred

        null_pred = self.backbone_with_pred_head(*args, drop_text_cond=True, **kwargs)

        return pred + (pred - null_pred) * cfg_strength

    def forward(
        self,
        input_features: torch.FloatTensor,
        text_input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Forward training

        Args:
            input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_size)`):
                Input feature.

            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding features. Mask values selected in `[0, 1]`:

                - 1 for **not masked**,
                - 0 for **masked**.

            text_input_ids (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`):
                Indices of input sequence text tokens in the vocabulary.  should
                you provide it.
        """
        batch, seq_len = input_features.shape[:2]
        device = input_features.device
        if attention_mask is None:
            attention_mask = torch.ones((batch, seq_len), device=device)

        # get a random span to mask out for training conditionally
        frac_lengths = (
            torch.zeros((batch,), device=self.device)
            .float()
            .uniform_(*self.config.frac_lengths_mask)
        )
        rand_span_mask = mask_from_frac_lengths(frac_lengths, attention_mask)

        # mel x1, gaussian noise x0
        x1 = input_features
        x0 = torch.randn_like(x1)

        # random timestep
        times = torch.rand((batch,), dtype=input_features.dtype, device=device)
        # use cosine timestep scheduler from cosyvoice.
        times = 1 - torch.cos(times * 0.5 * torch.pi)

        # sample xt (w in the paper), target flow
        t = times[..., None, None]

        w = (1 - (1 - self.config.sigma) * t) * x0 + t * x1
        flow = x1 - (1 - self.config.sigma) * x0

        # predict vector fields condition on masked mel
        cond = torch.where(
            rand_span_mask[..., None],
            torch.zeros_like(x1) + self.config.pad_feats_val,
            x1,
        )
        pred_v = self.backbone_with_pred_head(
            w,
            cond,
            times=times,
            text_input_ids=text_input_ids,
            mask=attention_mask,
        )

        # flow matching loss
        loss = F.mse_loss(pred_v, flow, reduction="none")
        loss = loss[rand_span_mask].mean()
        return loss, cond, pred_v

    def generate(
        self,
        input_features: torch.FloatTensor,
        gen_duration: Union[int, Tensor],
        *,
        text_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        steps=32,
        cfg_strength=1.0,  # they used a classifier free guidance strength of 1.
        max_duration=4096,  # the max tot duration
        odeint_kwargs=None,
    ):
        """ODE based sampling"""
        odeint_kwargs = odeint_kwargs or dict(atol=1e-5, rtol=1e-5, method="midpoint")
        batch, cond_seq_len, device = *input_features.shape[:2], input_features.device
        if attention_mask is None:
            attention_mask = torch.ones((batch, cond_seq_len), device=device)

        lens = attention_mask.sum(-1).long()
        durations = lens
        max_idx = durations.max()
        input_features = input_features[:, :max_idx, :]

        if text_input_ids is not None:
            text_lens = (text_input_ids != self.config.pad_text_token_id).sum(dim=-1)
            durations = torch.maximum(
                text_lens, durations
            )  # make sure lengths are at least those of the text characters

        # duration
        if isinstance(gen_duration, int):
            gen_duration = torch.full(
                (batch,), gen_duration, device=device, dtype=torch.long
            )
        elif isinstance(gen_duration, list):
            gen_duration = torch.tensor(gen_duration, device=device, dtype=torch.long)
        durations = torch.maximum(
            durations + 1, lens + gen_duration
        )  # just add one token so something is generated
        durations = durations.clamp(max=max_duration)
        max_duration = durations.amax()

        cond = F.pad(
            input_features,
            (0, 0, 0, max_duration - max_idx),
            value=self.config.pad_feats_val,
        )
        cond_mask = make_non_pad_mask(lens, max_duration)
        cond_mask = cond_mask[..., None]  # (B, T, 1)

        mask = make_non_pad_mask(durations)

        # neural ode
        # at each step, conditioning is fixed
        step_cond = torch.where(
            cond_mask, cond, torch.zeros_like(cond) + self.config.pad_feats_val
        )

        def fn(t, x):
            # predict flow

            return self.cfg_backbone_with_pred_head(
                x,
                step_cond,
                times=t,
                text_input_ids=text_input_ids,
                mask=mask,
                cfg_strength=cfg_strength,
            )

        y0 = torch.randn_like(cond)
        t = torch.linspace(0, 1, steps, device=self.device)

        trajectory = odeint(fn, y0, t, **odeint_kwargs)
        sampled = trajectory[-1]

        out = sampled

        out = torch.where(cond_mask, cond, out)

        return out, cond_mask.squeeze(-1), mask


class DurationPredictor(E2TTSBase):
    def __init__(
        self,
        config: DurPredConfig,
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.backbone = DurationBackBone(config=config)

        self.text_embedding = TextEmbedding(
            config.hidden_size // 2 if config.cond_text_net else config.hidden_size,
            config.vocab_size,
            config.pad_text_token_id,
        )

        self.to_pred = nn.Sequential(
            nn.Linear(config.hidden_size, 1, bias=False), nn.Softplus()
        )

        self.post_init()
        # apply special scaled init to the residual projections, GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("out_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers)
                )

    def forward(
        self,
        input_features: torch.FloatTensor,
        text_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Training duration predictor

        Args:
            input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_size)`):
                Input feature.

            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding features. Mask values selected in `[0, 1]`:

                - 1 for **not masked**,
                - 0 for **masked**.

            text_input_ids (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`):
                Indices of input sequence text tokens in the vocabulary.  should
                you provide it.
        """
        batch, seq_len, device = *input_features.shape[:2], input_features.device
        if attention_mask is None:
            attention_mask = torch.ones((batch, seq_len), device=device)

        lens = attention_mask.sum(-1).long()

        # random tail mask for regression
        rand_frac_index = input_features.new_zeros(batch).uniform_(0, 1)
        rand_index = (rand_frac_index * lens).long()
        text_lens = (text_input_ids != self.config.pad_text_token_id).sum(dim=-1)
        max_rand_idx = rand_index.max()
        input_features = input_features[:, :max_rand_idx, :]
        max_lens = torch.maximum(
            text_lens, rand_index
        )  # make sure lengths are at least those of the text characters
        max_length = max_lens.max()
        input_features = F.pad(
            input_features,
            (0, 0, 0, max_length - max_rand_idx),
            value=self.config.pad_feats_val,
        )
        # text
        text_embed = self.text_embedding(text_input_ids, max_length)

        mask = make_non_pad_mask(rand_index, max_length)
        input_features = input_features.masked_fill(
            ~(mask[..., None]), self.config.pad_feats_val
        )
        hidden_states = self.backbone(input_features, text_embed)
        mean = maybe_masked_mean(hidden_states, mask)  # (B, D)
        pred = self.to_pred(mean).squeeze(-1)  # (B,)
        return F.mse_loss(pred, lens.float())

    def generate(
        self,
        input_features: torch.FloatTensor,
        text_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Predict duration

        Args:
            input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_size)`):
                Input feature.

            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding features. Mask values selected in `[0, 1]`:

                - 1 for **not masked**,
                - 0 for **masked**.

            text_input_ids (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`):
                Indices of input sequence text tokens in the vocabulary.  should
                you provide it.
        """
        batch, seq_len, device = *input_features.shape[:2], input_features.device
        if attention_mask is None:
            attention_mask = torch.ones((batch, seq_len), device=device)

        lens = attention_mask.sum(-1).long()
        max_idx = lens.max()

        input_features = input_features[:, :max_idx, :]
        # text
        text_lens = (text_input_ids != self.config.pad_text_token_id).sum(dim=-1)
        max_lens = torch.maximum(
            text_lens, lens
        )  # make sure lengths are at least those of the text characters
        max_length = max_lens.max()
        input_features = F.pad(
            input_features,
            (0, 0, 0, max_length - max_idx),
            value=self.config.pad_feats_val,
        )
        # text
        text_embed = self.text_embedding(text_input_ids, max_length)

        mask = make_non_pad_mask(lens, max_length)
        input_features = input_features.masked_fill(
            ~(mask[..., None]), self.config.pad_feats_val
        )
        hidden_states = self.backbone(
            input_features,
            text_embed,
        )
        mean = maybe_masked_mean(hidden_states, mask)  # (B, D)
        pred = self.to_pred(mean).squeeze(-1)  # (B,)
        return pred


def maybe_masked_mean(
    inputs: Tensor,
    attention_mask: Optional[Tensor] = None,
):
    assert inputs.ndim == 3
    if attention_mask is None:
        return inputs.mean(dim=1)
    inputs = torch.where(attention_mask[..., None], inputs, torch.zeros_like(inputs))
    num = inputs.sum(dim=1)  # (B, D)
    den = attention_mask.sum(-1, keep_dim=True).float()  # (B, 1)
    return num / den.clamp(min=1.0)


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


def mask_from_frac_lengths(frac_lengths: Tensor, attn_mask=None, seq_len=None):
    assert frac_lengths.ndim == 1

    if (int(attn_mask is None) + int(seq_len is None)) != 1:
        raise ValueError(
            f"Please provide either attn_mask or seq_len, but got {attn_mask is None} and {seq_len is None}."
        )
    if attn_mask is not None:
        assert attn_mask.ndim == 2 and attn_mask.shape[0] == frac_lengths.shape[0]
        seq_len = attn_mask.sum(-1).long()
        mask_len = attn_mask.shape[1]
    else:
        seq_len = seq_len.long()
        mask_len = seq_len.max().item()
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = (
        torch.zeros_like(frac_lengths, device=frac_lengths.device)
        .float()
        .uniform_(0, 1)
    )
    start = (max_start * rand).clamp(min=0)
    end = start + lengths

    pad_mask = _pad_mask_from_start_end_indices(mask_len, start, end)
    if attn_mask is not None:
        pad_mask &= attn_mask.bool()
    return pad_mask


def _pad_mask_from_start_end_indices(max_seq_len: int, start: Tensor, end: Tensor):
    assert start.shape == end.shape
    device = start.device
    seq = torch.arange(max_seq_len, device=device, dtype=torch.long)
    seq = seq.reshape(*((-1,) * start.ndim), max_seq_len)
    seq = seq.expand(*start.shape, max_seq_len)

    mask = seq >= start[..., None].long()
    mask &= seq < end[..., None].long()
    return mask
