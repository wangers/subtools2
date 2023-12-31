# Copyright 3D-Speaker. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
CAM++ implementation modified from:
https://github.com/alibaba-damo-academy/3D-Speaker

refs:
[1] Hui Wang, Siqi Zheng, Yafeng Chen, Luyao Cheng and Qian Chen.
    "CAM++: A Fast and Efficient Network for Speaker Verification
    Using Context-Aware Masking". arXiv preprint arXiv:2303.00332
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import nn

from egrecho.core.model_base import ModelBase
from egrecho.models.campplus.campplus_config import CamPPConfig
from egrecho.nn.activation import Nonlinearity
from egrecho.nn.components import DenseLayer, TDNNBlock


def get_bn_relu(channels, eps=1e-5):
    return nn.Sequential(
        OrderedDict(
            [
                ("batchnorm", nn.BatchNorm1d(channels, eps=eps)),
                ("relu", Nonlinearity("relu")),
            ]
        )
    )


def statistics_pooling(
    x: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
    unbiased: bool = True,
):
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=unbiased)
    stats = torch.cat([mean, std], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


class StatsPool(nn.Module):
    def forward(self, x):
        return statistics_pooling(x)


class BasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=(stride, 1), padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FCM(nn.Module):
    def __init__(
        self, block=BasicResBlock, num_blocks=[2, 2], m_channels=32, feat_dim=80
    ):
        super(FCM, self).__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(
            1, m_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(m_channels)

        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[0], stride=2)

        self.conv2 = nn.Conv2d(
            m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))

        shape = out.shape
        out = out.reshape(shape[0], shape[1] * shape[2], shape[3])
        return out


class CAMLayer(nn.Module):
    def __init__(
        self,
        bn_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        bias,
        reduction=2,
    ):
        super(CAMLayer, self).__init__()
        self.linear_local = nn.Conv1d(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        context = self.relu(self.linear1(context))
        m = self.sigmoid(self.linear2(context))
        return y * m

    def seg_pooling(self, x, seg_len: int = 100, stype: str = "avg"):
        if stype == "avg":
            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        elif stype == "max":
            seg = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        else:
            raise ValueError("Wrong segment pooling type.")
        shape = seg.shape
        seg = (
            seg.unsqueeze(-1)
            .expand(shape[0], shape[1], shape[2], seg_len)
            .reshape(shape[0], shape[1], -1)
        )
        seg = seg[..., : x.shape[-1]]
        return seg


class CAMDenseTDNNLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bn_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        memory_efficient=False,
    ):
        super(CAMDenseTDNNLayer, self).__init__()
        assert (
            kernel_size % 2 == 1
        ), "Expect equal paddings, but got even kernel size ({})".format(kernel_size)
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_bn_relu(in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_bn_relu(bn_channels)
        self.cam_layer = CAMLayer(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def bn_function(self, x):
        return self.linear1(self.nonlinear1(x))

    @torch.jit.unused
    def bn_function_checkpointed(self, x: torch.Tensor) -> torch.Tensor:
        return cp.checkpoint(self.bn_function, x)

    def forward(self, x):
        # support torch.jit.script
        if self.training and self.memory_efficient:
            x = self.bn_function_checkpointed(x)
        else:
            x = self.bn_function(x)
        x = self.cam_layer(self.nonlinear2(x))
        return x


class CAMDenseTDNNBlock(nn.ModuleList):
    def __init__(
        self,
        num_layers,
        in_channels,
        out_channels,
        bn_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        memory_efficient=False,
    ):
        super(CAMDenseTDNNBlock, self).__init__()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                memory_efficient=memory_efficient,
            )
            self.add_module("tdnnd%d" % (i + 1), layer)

    def forward(self, x):
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(TransitLayer, self).__init__()
        self.nonlinear = get_bn_relu(in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.nonlinear(x)
        x = self.linear(x)
        return x


class CamBase(ModelBase):
    """
    Abstract base module for ecapa.
    """

    main_input_name = ["input_features"]

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class CamPP(CamBase):
    """Cam++ xvector.

    Adapts DenseNet-based Tdnn to sv task, key features:
        - Employing a two-dimensional front-end convolution (FCM) to extract a
        comprehensive representation from audio data.
        - Preferring a depth-first design over a width-first.
        - Introducing a Context-aware Masking (CAM) module after each block, enabling the aggregation of local and
        global contextual information through pooling and attention mechanisms.
    """

    def __init__(
        self,
        config: CamPPConfig,
    ):
        super(CamPP, self).__init__()

        self.head = FCM(feat_dim=config.inputs_dim)
        channels = self.head.out_channels
        init_channels = config.init_channels

        self.xvector = nn.Sequential(
            OrderedDict(
                [
                    (
                        "tdnn",
                        TDNNBlock(
                            channels,
                            init_channels,
                            5,
                            stride=2,
                            dilation=1,
                            pre_norm=True,
                        ),
                    ),
                ]
            )
        )
        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(
            zip((12, 24, 16), (3, 3, 3), (1, 2, 2))
        ):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=config.growth_rate,
                bn_channels=config.bn_size * config.growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                memory_efficient=config.memory_efficient,
            )
            self.xvector.add_module("block%d" % (i + 1), block)
            channels = channels + num_layers * config.growth_rate
            self.xvector.add_module(
                "transit%d" % (i + 1),
                TransitLayer(channels, channels // 2, bias=False),
            )
            channels //= 2

        self.xvector.add_module("out_nonlinear", get_bn_relu(channels))

        self.xvector.add_module("stats", StatsPool())

        self.xvector.add_module("dense", DenseLayer(channels * 2, config.embd_dim))

        self.post_init()

    def forward(self, input_features: torch.Tensor):
        x = input_features.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)
        x = self.xvector(x)
        return x
