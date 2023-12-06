# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2022-05-27)
# refs:
# 1.  ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification
#           https://arxiv.org/abs/2005.07143

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from egrecho.core.model_base import ModelBase
from egrecho.models.ecapa.ecapa_config import EcapaConfig
from egrecho.nn.components import TDNNBlock


def compute_statistics(x, m, dim: int = -1, stddev: bool = True, eps: float = 1e-5):
    """
    Calculate the mean and optionally the standard deviation of the given input data.

    Cases:
    - Case 1 (mean-var pooling): For the masked input 'm', we assume that each unmasked value (i.e., 1)
      in the time dimension is scaled by 'VALID_LEN / TOTAL_LEN' for each sample in the batch.
    - Case 2 (asp pooling): To calculate `alpha`, mask the value in invalid time to -inf before calculation is ok.

    See: :method::`MQMHASP.forward`


    Args:
        x (Tensor):
            Input data tensor.
        m (Tensor [B, ...]):
            A scaled mask representing which values to include in the calculation.
        dim (int, optional):
            Dimension along which to compute statistics. Default is -1, meaning computation along the last dimension.
        stddev (bool, optional):
            Whether to calculate the standard deviation. If True, it computes the standard deviation;
            otherwise, the standard deviation will be an empty tensor. Default is True.
        eps (float, optional):
            Small constant for stable computation of the standard deviation. Default is 1e-5.

    Returns:
        Tuple[Tensor, Tensor]:
            A tuple containing the mean and standard deviation. If stddev is False,
            the standard deviation will be an empty tensor.

    Note:
        - This function is used to compute the mean and optional standard deviation of the data specified by
          the mask 'm' on the given data tensor 'x'.
        - If stddev is True, it calculates the standard deviation;
          otherwise, the standard deviation will be an empty tensor.
    """
    mean = torch.sum(m * x, dim=dim, keepdim=True)

    if stddev:
        # std = torch.sqrt(
        #     (m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps)
        # )
        std = torch.sqrt(
            (torch.sum(m * (x**2), dim=dim, keepdim=True) - mean**2).clamp_(eps)
        )
    else:
        std = torch.empty(0)
    return mean, std


class MQMHASP(nn.Module):
    """
    Reference:
       Miao Zhao, Yufeng Ma, and Yiwei Ding et al. "Multi-query multi-head attention pooling and Inter-topK penalty for speaker verification".
       https://arxiv.org/pdf/2110.05042.pdf
    """

    def __init__(
        self,
        in_dim,
        num_q: int = 2,
        num_head: int = 4,
        hidden_size: int = 128,
        stddev: bool = True,
        share: bool = False,
        affine_layers: int = 2,
        time_attention=False,
        norm_type: Literal["bn", "ln"] = "bn",
    ):
        super(MQMHASP, self).__init__()
        self.stddev = stddev
        # self.output_dim = in_dim*2 if self.stddev else in_dim
        self.num_head = max(1, num_head)
        self.num_q = max(1, num_q)
        self.time_attention = time_attention
        assert (in_dim % num_head) == 0
        att_idim = in_dim // num_head
        if time_attention:
            att_idim = (in_dim * 3) // num_head if stddev else (in_dim * 2) // num_head
        att_odim = 1 if share else in_dim // num_head
        self.attention = self.build_attention(
            att_idim * num_head,
            att_odim * num_head * num_q,
            num_q,
            num_head,
            affine_layers,
            hidden_size,
            norm_type=norm_type,
        )
        self.out_dim = in_dim * num_q * 2 if stddev else in_dim * num_q

    def forward(self, x, mask: torch.Tensor = torch.ones((0, 0, 0))):
        """
        x: input feature [B, F, T]
        returns: pooling statiscs [B, F * qs, 1]
        """
        B, C, T = x.shape

        if mask.size(2) == 0:
            mask = torch.ones((B, 1, T)).to(x.device)

        if self.time_attention:
            total = mask.sum(dim=2, keepdim=True)  # [B, *, 1]
            mean, std = compute_statistics(x, mask / total, stddev=self.stddev)
            mean = (mean.repeat(1, 1, T)).view(B, self.num_head, -1, T)
            x_in = x.view(B, self.num_head, -1, T)
            if self.stddev:
                std = (std.repeat(1, 1, T)).view(B, self.num_head, -1, T)
                x_in = torch.cat([x_in, mean, std], dim=2)
            else:
                x_in = torch.cat([x_in, mean], dim=2)
            x_in = x_in.reshape(B, -1, T)
        else:
            x_in = x
        alpha = self.attention(x_in)  # [B, head * att_dim, T]

        alpha = alpha.masked_fill(mask == 0, float("-inf"))

        alpha = F.softmax(alpha, dim=2)
        alpha = alpha.reshape(B, self.num_head, self.num_q, -1, T)

        mean, std = compute_statistics(
            x.reshape(B, self.num_head, 1, -1, T), alpha, stddev=self.stddev
        )  # mean: [B, head, q, C/head, 1]
        mean = mean.reshape(B, -1, 1)
        if self.stddev:
            std = std.reshape(B, -1, 1)
            out = torch.cat([mean, std], dim=1)
        else:
            out = mean
        return out

    def get_output_dim(self):
        return self.out_dim

    def build_attention(
        self,
        idim,
        odim,
        num_q,
        num_head,
        affine_layers=1,
        hidden_size=128,
        norm_type="bn",
    ):
        assert affine_layers in [1, 2], "Expected 1 or 2 affine layers."
        assert (idim % num_head) == 0
        assert (odim % (num_head * num_q)) == 0

        if affine_layers == 2:
            if norm_type == "bn":
                norm = torch.nn.BatchNorm1d(hidden_size * num_head * num_q)
            elif norm_type == "ln":
                norm = torch.nn.GroupNorm(
                    num_head * num_q, hidden_size * num_head * num_q
                )
            else:
                raise ValueError("Unsupport norm type:{}".format(norm_type))
            att = torch.nn.Sequential(
                torch.nn.Conv1d(
                    idim, hidden_size * num_head * num_q, kernel_size=1, groups=num_head
                ),
                torch.nn.ReLU(),
                norm,
                torch.nn.Tanh(),
                torch.nn.Conv1d(
                    hidden_size * num_head * num_q,
                    odim,
                    kernel_size=1,
                    groups=num_head * num_q,
                ),
            )
        elif affine_layers == 1:
            att = torch.nn.Conv1d(idim, odim, kernel_size=1, groups=num_head)

        else:
            raise ValueError(
                "Expected 1 or 2 affine layers, but got {}.".format(affine_layers)
            )
        return att

    def extra_repr(self):
        return "(stddev={stddev}, num_head={num_head}, num_q={num_q}, out_dim={out_dim}) ".format(
            **self.__dict__
        )


# class LayerNorm(torch.nn.LayerNorm):
#     """Layer normalization module

#     :param int nout: output dim size
#     :param int dim: dimension to be normalized
#     """

#     def __init__(self, nout, dim=-1, eps=1e-5, learnabel_affine: bool = True):
#         super(LayerNorm, self).__init__(
#             nout, eps=eps, elementwise_affine=learnabel_affine
#         )
#         self.dim = dim

#     def forward(self, x):
#         """Apply layer normalization

#         :param torch.Tensor x: input tensor
#         :return: layer normalized tensor
#         :rtype torch.Tensor
#         """
#         if self.dim == -1:
#             return F.layer_norm(
#                 x, self.normalized_shape, self.weight, self.bias, self.eps
#             )
#         return F.layer_norm(
#             x.transpose(1, -1), self.normalized_shape, self.weight, self.bias, self.eps
#         ).transpose(1, -1)


# class TDNNBlock(nn.Module):
#     r"""
#     A simplified version of `ReluBatchNormTdnnLayer` that represents a block with Convolution-Relu-BatchNormalization (BN).

#     This block is designed to work with 1D convolution using `torch.nn.Conv1d` by default. It enforces an odd kernel size for easy padding.

#     Args:
#         in_channels (int):
#             The number of input channels.
#         out_channels (int):
#             The number of output channels.
#         kernel_size (int):
#             The size of the convolutional kernel. Defaults is 1.
#         stride (int, optional):
#             The stride value for the convolutional layer. Default is 1.
#         pad (bool, optional):
#             Determines whether to pad the input. If True, the kernel size must be odd. Default is True.
#         dilation (int, optional):
#             The dilation rate for the convolutional layer. Default is 1.
#         bias (bool, optional):
#             Determines whether to include bias in the convolutional layer. Default is False.
#         affine (bool, optional):
#             Determines whether to apply learnable affine transformations in Batch Normalization. Default is True.
#         norm_type (Literal['bn', 'ln'], optional):
#             Specifies the type of normalization to use.
#             ('bn' for Batch Normalization, 'ln' for Layer Normalization). Default is 'bn'.
#         pre_norm (bool, optional):
#             Whether applys normalization before the activation function. Default is False.

#     Note:
#         - This class simplifies the architecture of a TDNN (Time Delay Neural Network) layer
#         by combining Convolution, ReLU activation, and Batch Normalization (or Layer Normalization).
#         - When `pad` is True, it enforces an odd kernel size for easy padding.
#     """

#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size: int = 1,
#         stride: int = 1,
#         pad: bool = True,
#         dilation: int = 1,
#         bias: bool = True,
#         affine: bool = True,
#         norm_type: Literal["bn", "ln"] = "bn",
#         pre_norm: bool = False,
#     ):
#         super(TDNNBlock, self).__init__()
#         if pad:
#             assert (
#                 kernel_size % 2 == 1
#             ), f"Expect equal paddings, but got even kernel size ({kernel_size})."

#             padding = (kernel_size - 1) // 2 * dilation
#         else:
#             padding = 0
#         self.linear = nn.Conv1d(
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride=stride,
#             padding=padding,
#             dilation=dilation,
#             bias=bias,
#         )
#         assert norm_type in (
#             "bn",
#             "ln",
#         ), f"Expect norm of 'bn' or 'ln', but got {norm_type}"
#         norm_mod = (
#             nn.BatchNorm1d(out_channels, affine=affine)
#             if norm_type == "bn"
#             else LayerNorm(out_channels, learnabel_affine=affine)
#         )

#         self.nonlinear = (
#             nn.Sequential(
#                 norm_mod,
#                 nn.ReLU(inplace=True),
#             )
#             if pre_norm
#             else nn.Sequential(
#                 nn.ReLU(inplace=True),
#                 norm_mod,
#             )
#         )

#     def forward(self, x: torch.Tensor):
#         unsqueezed = False
#         if x.ndim == 2:
#             x = x.unsqueeze(dim=2)
#             unsqueezed = True
#         x = self.linear(x)
#         x = self.nonlinear(x)

#         return x.squeeze(2) if unsqueezed else x


class Res2NetBlock(nn.Module):
    """An implementation of Res2NetBlock w/ dilation.

    Args:
        channels : int
            The number of channels expected in the input.
        scale : int
            The scale of the Res2Net block.
        kernel_size: int
            The kernel size of the Res2Net block.
        dilation : int
            The dilation of the Res2Net block.

    Example:
        >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
        >>> layer = Res2NetBlock(64, scale=4, dilation=3)
        >>> out_tensor = layer(inp_tensor).transpose(1, 2)
        >>> out_tensor.shape
        torch.Size([8, 120, 64])
    """

    def __init__(
        self,
        channels,
        scale=8,
        kernel_size=3,
        dilation=1,
        bias=True,
    ):
        super(Res2NetBlock, self).__init__()
        assert channels % scale == 0
        assert scale > 1
        hidden_channel = channels // scale

        self.blocks = nn.ModuleList(
            [
                TDNNBlock(
                    hidden_channel,
                    hidden_channel,
                    kernel_size,
                    dilation=dilation,
                    bias=bias,
                )
                for _ in range(scale - 1)
            ]
        )

        self.scale = scale

    def forward(self, x):
        y = []
        spx = torch.chunk(x, self.scale, dim=1)
        sp = spx[0]
        y.append(sp)
        for i, block in enumerate(self.blocks):
            if i == 0:
                sp = spx[i + 1]
            if i >= 1:
                sp = sp + spx[i + 1]
            sp = block(sp)
            y.append(sp)

        y = torch.cat(y, dim=1)
        return y


class SE_Connect(nn.Module):
    """The SE connection of 1D case."""

    def __init__(self, channels, bottleneck=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, bottleneck)
        self.linear2 = nn.Linear(bottleneck, channels)

    def forward(self, x: Tensor):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)

        return out


class SE_Res2Block(nn.Module):
    """SE-Res2Block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        scale=8,
    ):
        super(SE_Res2Block, self).__init__()
        width = math.floor(in_channels / scale)

        self.conv_relu_bn1 = TDNNBlock(in_channels, width * scale)

        self.res2net_block = Res2NetBlock(
            width * scale,
            scale=scale,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.conv_relu_bn2 = TDNNBlock(width * scale, out_channels)
        self.se = SE_Connect(out_channels)
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        x = self.conv_relu_bn1(x)

        x = self.res2net_block(x)

        x = self.conv_relu_bn2(x)
        x = self.se(x)
        return x + residual


class EcapaBase(ModelBase):
    """
    Abstract base module for ecapa.
    """

    main_input_name = ["input_features"]

    def _init_weights(self, module):
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)

            if module.bias is not None:
                k = math.sqrt(
                    module.groups / (module.in_channels * module.kernel_size[0])
                )
                nn.init.uniform_(module.bias, a=-k, b=k)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class EcapaXvector(EcapaBase):
    """Implementation of ecapa-tdnn.

    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification".

    Note that concatenates the last frame-wise layer with non-weighted mean and standard deviation,
    will bring little improvment but significantly increases model parameters.
    """

    def __init__(
        self,
        config: EcapaConfig,
    ):
        super().__init__()
        self.config = config

        self.embd_dim = config.embd_dim
        channels = config.channels
        self.dummy_inputs = {
            "input_features": torch.randn(2, 200, self.config.inputs_dim)
        }
        self.layer1 = TDNNBlock(config.inputs_dim, channels, kernel_size=5)
        self.layer2 = SE_Res2Block(
            channels, channels, kernel_size=3, dilation=2, scale=8
        )
        self.layer3 = SE_Res2Block(
            channels, channels, kernel_size=3, dilation=3, scale=8
        )
        self.layer4 = SE_Res2Block(
            channels, channels, kernel_size=3, dilation=4, scale=8
        )
        cat_channels = channels * 3
        mfa_dim = config.mfa_dim
        self.mfa = TDNNBlock(cat_channels, mfa_dim)

        pooling_params = config.pooling_params
        self.stats: MQMHASP = MQMHASP(mfa_dim, **pooling_params)
        self.bn_stats = nn.BatchNorm1d(self.stats.get_output_dim())
        self.embd_layer_num = config.embd_layer_num
        if self.embd_layer_num == 1:
            self.embd1 = nn.Linear(self.stats.get_output_dim(), self.embd_dim)
            self.embd2 = nn.Identity()
        else:
            self.embd1 = TDNNBlock(self.stats.get_output_dim(), self.embd_dim)
            self.embd2 = nn.Linear(self.embd_dim, self.embd_dim)
        if config.post_norm:
            self.post_norm = nn.BatchNorm1d(self.embd_dim)
        else:
            self.post_norm = nn.Identity()
        self.post_init()

    def forward(self, input_features: Tensor):
        x = input_features.permute(0, 2, 1)  # [B, T, F] -> [B, F, T]
        x = self.layer1(x)
        x1 = self.layer2(x)
        x2 = self.layer3(x + x1)
        x3 = self.layer4(x + x1 + x2)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.mfa(x)

        x = self.bn_stats(self.stats(x))
        x = x.squeeze(2)
        if self.embd_layer_num == 1:
            embd_far = torch.empty(0)  # jit compatible
            embd = self.embd1(x)
        else:
            embd_far = self.embd1(x)
            embd = self.embd2(embd_far)
        embd = self.post_norm(embd)

        return embd, embd_far
