# MIT License

# Copyright (c) 2023 Phil Wang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright (Leo 2024-08)

"""SimpleGateLoopLayer refer to lucidrains (Expermental).
    https://github.com/lucidrains/gateloop-transformer
"""

from functools import partial
from typing import Callable, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from egrecho.utils.common import alt_none
from egrecho.utils.imports import RequirementCache

if not (re_eins := RequirementCache('einops')):
    raise ImportError(f'{re_eins}')
from einops import pack, unpack
from einops.layers.torch import Rearrange


def eps_by_dtype(dtype):
    return 1e-7 if dtype == torch.float16 else 1e-20


def abs_clamp_eps(t, eps=None):
    eps = alt_none(eps, eps_by_dtype(t.dtype))
    sign = torch.sign(t)
    return sign * t.abs().clamp(min=eps)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


# coppied from https://github.com/lucidrains/gateloop-transformer/blob/main/gateloop_transformer/simplified_gate_loop.py
class SimpleGateLoopLayer(nn.Module):
    """
    simplified gate loop
    seeing if it can supplement attention as shown in https://github.com/lucidrains/mega-pytorch
    """

    def __init__(
        self, dim, prenorm=True, use_heinsen=False, post_ln=False, reverse=False
    ):
        super().__init__()
        self.norm = RMSNorm(dim) if prenorm else nn.Identity()

        self.dim = dim

        self.to_qkva = nn.Sequential(
            nn.Linear(dim, dim * 3, bias=False),
            Rearrange('b n (qkva d) -> qkva (b d) n 1', qkva=3),
        )

        self.use_heinsen = use_heinsen

        if use_heinsen:
            self.gate_loop_fn = partial(gate_loop_operator, heinsen=True)
        else:
            self.gate_loop_fn = gate_loop_operator

        self.maybe_post_ln = nn.LayerNorm(dim) if post_ln else nn.Identity()
        self.split_heads = Rearrange('(b d) n 1 -> b n d', d=dim)

        self.reverse = reverse

    def forward(self, x, cache=None, return_cache=False):
        if self.reverse:
            x = torch.flip(x, dims=(-2,))

        x = self.norm(x)

        q, kv, a = self.to_qkva(x)

        out, cache = self.gate_loop_fn(q, kv, a.sigmoid(), cache=cache)

        out = self.split_heads(out)
        out = self.maybe_post_ln(out)

        if self.reverse:
            out = torch.flip(out, dims=(-2,))

        if not return_cache:
            return out

        assert not self.reverse, 'caching only works with non-reversed seq'

        return out, cache


# associative scan using heinsen sequences
# https://github.com/glassroom/heinsen_sequence
# graciously shared to the world by Franz A. Heinsen in https://arxiv.org/abs/2311.06281 in October 2023
def heinsen_associative_scan(a, kv, eps=None):
    eps = eps or eps_by_dtype(a.dtype)
    log_a = a.clamp(min=eps).log()

    log_kv = abs_clamp_eps(kv).to(dtype=torch.complex64).log()

    a_star = torch.cumsum(log_a, dim=1)
    log_x0_plus_b_star = torch.logcumsumexp(log_kv - a_star, dim=1)
    log_x = a_star + log_x0_plus_b_star
    return a_star.exp().real, log_x.exp().real


# naive associative scan with some torchscript of binary operator


@torch.jit.script
def binary_operator(a: Tuple[Tensor, Tensor], b: Tuple[Tensor, Tensor]):
    a_i, kv_i = a
    a_j, kv_j = b
    return a_j * a_i, torch.addcmul(kv_j, a_j, kv_i)


def gate_loop_operator(q, kv, a, cache=None, heinsen=False):

    if cache:
        cache_a, cache_kv = cache
        a, a_ps = pack([cache_a, a], 'b * d')
        kv, kv_ps = pack([cache_kv, kv], 'b * d')

    if heinsen:
        a, kv = heinsen_associative_scan(a, kv)
    else:
        a, kv = associative_scan(binary_operator, (a, kv))

    if cache:
        _, a = unpack(a, a_ps, 'b * d')
        _, kv = unpack(kv, kv_ps, 'b * d')

    return q * kv, (a[:, -1], kv[:, -1])


def pad_at_dim(t, pad, dim=-1, value=0.0):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), value=value)


# Pytorch impl. of jax.lax.associative_scan
# made specifically for axis of 1 (sequence of tokens for autoregressive modeling)


def associative_scan(operator: Callable, elems: Tuple[Tensor, Tensor]):
    num_elems = int(elems[0].shape[1])

    if not all(int(elem.shape[1]) == num_elems for elem in elems[1:]):
        raise ValueError(
            'Array inputs to associative_scan must have the same '
            'first dimension. (saw: {})'.format([elem.shape for elem in elems])
        )

    def _scan(elems):
        """Perform scan on `elems`."""
        num_elems = elems[0].shape[1]

        if num_elems < 2:
            return elems

        # Combine adjacent pairs of elements.

        reduced_elems = operator(
            [elem[:, :-1:2] for elem in elems], [elem[:, 1::2] for elem in elems]
        )

        # Recursively compute scan for partially reduced tensors.

        odd_elems = _scan(reduced_elems)

        if num_elems % 2 == 0:
            even_elems = operator(
                [e[:, :-1] for e in odd_elems], [e[:, 2::2] for e in elems]
            )
        else:
            even_elems = operator(odd_elems, [e[:, 2::2] for e in elems])

        # The first element of a scan is the same as the first element
        # of the original `elems`.

        even_elems = [
            torch.cat([elem[:, :1], result], dim=1)
            for (elem, result) in zip(elems, even_elems)
        ]

        return list(map(_interleave, even_elems, odd_elems))

    return _scan(elems)


def _interleave(a, b):
    a_axis_len, b_axis_len = a.shape[1], b.shape[1]
    output_axis_len = a_axis_len + b_axis_len

    if a_axis_len == (b_axis_len + 1):
        b = pad_at_dim(b, (0, 1), dim=1)

    stacked = torch.stack([a, b], dim=2)
    interleaved = torch.flatten(stacked, start_dim=1, end_dim=2)

    return interleaved[:, :output_axis_len]
