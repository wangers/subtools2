# Copyright xmuspeech (Author: Leo 2022-05-27)
# refs:
# 1.  ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification
#           https://arxiv.org/abs/2005.07143

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from egrecho.core.model_base import ModelBase
from egrecho.models.ecapa.ecapa_config import EcapaConfig
from egrecho.nn.components import TDNNBlock
from egrecho.nn.pooling import MQMHASP


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

    Note that we DON'T concatenate the last frame-wise layer with non-weighted mean and standard deviation,
    because it brings little improvment but significantly increases model parameters.
    As a result, this implementation basically equals the A.2 of Table 2 in the paper.
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
            "input_features": torch.randn(2, self.config.inputs_dim, 200)
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
        self.stats: MQMHASP(mfa_dim, **pooling_params)
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

    def forward(self, input_features: Tensor, position: str = "near") -> Tensor:
        x = input_features.permute(0, 2, 1)
        x = self.layer1(x)
        x1 = self.layer2(x)
        x2 = self.layer3(x + x1)
        x3 = self.layer4(x + x1 + x2)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.mfa(x)

        x = self.bn_stats(self.stats(x))
        x = x.squeeze(2)
        if self.embd_layer_num == 1:
            if position == "far":
                raise RuntimeError(
                    "Request embd in far positon, but got one embd layer related to near."
                )
            embd_far = torch.empty(0)  # jit compatible
            embd = self.embd1(x)
        else:
            embd_far = self.embd1(x)
            embd = self.embd2(embd_far)
        embd = self.post_norm(embd)

        return embd
