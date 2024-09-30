import math
from typing import Optional, Union, Sequence

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule
from .conv import Conv



def autopad(kernel_size: int, padding: int = None, dilation: int = 1):
    assert kernel_size % 2 == 1, 'if use autopad, kernel size must be odd'
    if dilation > 1:
        kernel_size = dilation * (kernel_size - 1) + 1
    if padding is None:
        padding = kernel_size // 2
    return padding


def make_divisible(value, divisor, min_value=None, min_ratio=0.9):
    """Make divisible function.

    This function rounds the channel number to the nearest value that can be
    divisible by the divisor. It is taken from the original tf repo. It ensures
    that all layers have a channel number that is divisible by divisor. It can
    be seen here: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py  # noqa

    Args:
        value (int, float): The original channel number.
        divisor (int): The divisor to fully divide the channel number.
        min_value (int): The minimum value of the output channel.
            Default: None, means that the minimum value equal to the divisor.
        min_ratio (float): The minimum ratio of the rounded channel number to
            the original channel number. Default: 0.9.

    Returns:
        int: The modified output channel number.
    """

    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than (1-min_ratio).
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value


class BCHW2BHWC(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return x.permute([0, 2, 3, 1])


class BHWC2BCHW(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return x.permute([0, 3, 1, 2])


class GSiLU(BaseModule):
    """Global Sigmoid-Gated Linear Unit, reproduced from paper <SIMPLE CNN FOR VISION>"""
    def __init__(self):
        super().__init__()
        self.adpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return x * torch.sigmoid(self.adpool(x))


class CAA(BaseModule):
    """Context Anchor Attention"""
    def __init__(
            self,
            channels: int,
            h_kernel_size: int = 11,
            v_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1,
                                 (0, h_kernel_size // 2), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1,
                                 (v_kernel_size // 2, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.conv2 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor


class ConvFFN(BaseModule):
    """Multi-layer perceptron implemented with ConvModule"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            hidden_channels_scale: float = 4.0,
            hidden_kernel_size: int = 3,
            dropout_rate: float = 0.,
            add_identity: bool = True,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        out_channels = out_channels or in_channels
        hidden_channels = int(in_channels * hidden_channels_scale)

        self.ffn_layers = nn.Sequential(
            BCHW2BHWC(),
            nn.LayerNorm(in_channels),
            BHWC2BCHW(),
            ConvModule(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(hidden_channels, hidden_channels, kernel_size=hidden_kernel_size, stride=1,
                       padding=hidden_kernel_size // 2, groups=hidden_channels,
                       norm_cfg=norm_cfg, act_cfg=None),
            GSiLU(),
            nn.Dropout(dropout_rate),
            ConvModule(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            nn.Dropout(dropout_rate),
        )
        self.add_identity = add_identity

    def forward(self, x):
        x = x + self.ffn_layers(x) if self.add_identity else self.ffn_layers(x)
        return x


class DownSamplingLayer(BaseModule):
    """Down sampling layer"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        out_channels = out_channels or (in_channels * 2)

        self.down_conv = ConvModule(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                    norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        return self.down_conv(x)


class InceptionBottleneck(BaseModule):   # KP Imodel
    """Bottleneck with Inception module"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
            dilations: Sequence[int] = (1, 1, 1, 1, 1),
            expansion: float = 1.0,
            add_identity: bool = True,
            with_caa: bool = True,
            caa_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        out_channels = out_channels or in_channels
        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        self.pre_conv = ConvModule(in_channels, hidden_channels, 1, 1, 0, 1,
                                   norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.dw_conv = ConvModule(hidden_channels, hidden_channels, kernel_sizes[0], 1,
                                  autopad(kernel_sizes[0], None, dilations[0]), dilations[0],
                                  groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv1 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[1], 1,
                                   autopad(kernel_sizes[1], None, dilations[1]), dilations[1],
                                   groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv2 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[2], 1,
                                   autopad(kernel_sizes[2], None, dilations[2]), dilations[2],
                                   groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv3 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[3], 1,
                                   autopad(kernel_sizes[3], None, dilations[3]), dilations[3],
                                   groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv4 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[4], 1,
                                   autopad(kernel_sizes[4], None, dilations[4]), dilations[4],
                                   groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.pw_conv = ConvModule(hidden_channels, hidden_channels, 1, 1, 0, 1,
                                  norm_cfg=norm_cfg, act_cfg=act_cfg)

        if with_caa:
            self.caa_factor = CAA(hidden_channels, caa_kernel_size, caa_kernel_size, None, None)
        else:
            self.caa_factor = None

        self.add_identity = add_identity and in_channels == out_channels

        self.post_conv = ConvModule(hidden_channels, out_channels, 1, 1, 0, 1,
                                    norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        x = self.pre_conv(x)

        y = x  # if there is an inplace operation of x, use y = x.clone() instead of y = x
        x = self.dw_conv(x)
        x = x + self.dw_conv1(x) + self.dw_conv2(x) + self.dw_conv3(x) + self.dw_conv4(x)
        x = self.pw_conv(x)
        if self.caa_factor is not None:
            y = self.caa_factor(y)
        if self.add_identity:
            y = x * y
            x = x + y
        else:
            x = x * y

        x = self.post_conv(x)
        return x


class PKIBlock(BaseModule):
    """Poly Kernel Inception Block"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
            dilations: Sequence[int] = (1, 1, 1, 1, 1),
            with_caa: bool = True,
            caa_kernel_size: int = 11,
            expansion: float = 1.0,
            ffn_scale: float = 4.0,
            ffn_kernel_size: int = 3,
            dropout_rate: float = 0.,
            drop_path_rate: float = 0.,
            layer_scale: Optional[float] = 1.0,
            add_identity: bool = True,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        out_channels = out_channels or in_channels
        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        if norm_cfg is not None:
            self.norm1 = build_norm_layer(norm_cfg, in_channels)[1]
            self.norm2 = build_norm_layer(norm_cfg, hidden_channels)[1]
        else:
            self.norm1 = nn.BatchNorm2d(in_channels)
            self.norm2 = nn.BatchNorm2d(hidden_channels)

        self.block = InceptionBottleneck(in_channels, hidden_channels, kernel_sizes, dilations,
                                         expansion=1.0, add_identity=True,
                                         with_caa=with_caa, caa_kernel_size=caa_kernel_size,
                                         norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.ffn = ConvFFN(hidden_channels, out_channels, ffn_scale, ffn_kernel_size, dropout_rate, add_identity=False,
                           norm_cfg=None, act_cfg=None)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        self.layer_scale = layer_scale
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(hidden_channels), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(out_channels), requires_grad=True)
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x):
        if self.layer_scale:
            if self.add_identity:
                x = x + self.drop_path(self.gamma1.unsqueeze(-1).unsqueeze(-1) * self.block(self.norm1(x)))
                x = x + self.drop_path(self.gamma2.unsqueeze(-1).unsqueeze(-1) * self.ffn(self.norm2(x)))
            else:
                x = self.drop_path(self.gamma1.unsqueeze(-1).unsqueeze(-1) * self.block(self.norm1(x)))
                x = self.drop_path(self.gamma2.unsqueeze(-1).unsqueeze(-1) * self.ffn(self.norm2(x)))
        else:
            if self.add_identity:
                x = x + self.drop_path(self.block(self.norm1(x)))
                x = x + self.drop_path(self.ffn(self.norm2(x)))
            else:
                x = self.drop_path(self.block(self.norm1(x)))
                x = self.drop_path(self.ffn(self.norm2(x)))
        return x


class PKIStage(BaseModule):
    """Poly Kernel Inception Stage"""
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_blocks: int,
            kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
            dilations: Sequence[int] = (1, 1, 1, 1, 1),
            expansion: float = 0.5,
            ffn_scale: float = 4.0,
            ffn_kernel_size: int = 3,
            dropout_rate: float = 0.,
            drop_path_rate: Union[float, list] = 0.,
            layer_scale: Optional[float] = 1.0,
            shortcut_with_ffn: bool = True,
            shortcut_ffn_scale: float = 4.0,
            shortcut_ffn_kernel_size: int = 5,
            add_identity: bool = True,
            with_caa: bool = True,
            caa_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        self.downsample = DownSamplingLayer(in_channels, out_channels, norm_cfg, act_cfg)

        self.conv1 = ConvModule(out_channels, 2 * hidden_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(2 * hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv3 = ConvModule(out_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.ffn = ConvFFN(hidden_channels, hidden_channels, shortcut_ffn_scale, shortcut_ffn_kernel_size, 0.,
                           add_identity=True, norm_cfg=None, act_cfg=None) if shortcut_with_ffn else None

        self.blocks = nn.ModuleList([
            PKIBlock(hidden_channels, hidden_channels, kernel_sizes, dilations, with_caa,
                     caa_kernel_size+2*i, 1.0, ffn_scale, ffn_kernel_size, dropout_rate,
                     drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                     layer_scale, add_identity, norm_cfg, act_cfg) for i in range(num_blocks)
        ])

    def forward(self, x):
        x = self.downsample(x)

        x, y = list(self.conv1(x).chunk(2, 1))
        if self.ffn is not None:
            x = self.ffn(x)

        z = [x]
        t = torch.zeros(y.shape, device=y.device)
        for block in self.blocks:
            t = t + block(y)
        z.append(t)
        z = torch.cat(z, dim=1)
        z = self.conv2(z)
        z = self.conv3(z)

        return z


class C2f_PKIBlock(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        # hidden_channels = make_divisible(int(c2 * expansion), 8)
        self.m = nn.ModuleList(PKIBlock(self.c, self.c) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))



# Ultralytics YOLO 🚀, AGPL-3.0 license
# YOLOv8 object detection model with P3-P5 outputs. For Usage examples see https://docs.ultralytics.com/tasks/detect

# Parameters
nc: 80 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.33, 0.25, 1024] # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
  s: [0.33, 0.50, 1024] # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
  m: [0.67, 0.75, 768] # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
  l: [1.00, 1.00, 512] # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
  x: [1.00, 1.25, 512] # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs

# YOLOv8.0n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2     1x3x640x640 -> 1x16x320x320
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4   1x16x320x320 -> 1x32x160x160
  - [-1, 3, C2f_PKIBlock, [128, True]]      #1x32x160x160 -> 1x32x160x160
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8    1x32x160x160 -> 1x64x80x80
  - [-1, 6, C2f, [256, True]]              # 1x64x80x80 -> 1x64x80x80
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16    1x64x80x80 -> 1x128x40x40
  - [-1, 6, C2f, [512, True]]             #1x128x40x40-> 1x128x40x40
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32  1x128x40x40-> 1x256x20x20
  - [-1, 3, C2f, [1024, True]]            # 1x256x20x20-> 1x256x20x20
  - [-1, 1, SPPF, [1024, 5]] # 9             1x256x20x20-> 1x256x20x20

# YOLOv8.0n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 1x256x40x40
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4  # # 1x384x40x40
  - [-1, 3, C2f, [512]] # 12                       1x128x40x40

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] #   1x128x80x80
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3    1x192x80x80
  - [-1, 3, C2f, [256]] # 15 (P3/8-small)          1x64x80x80

  - [-1, 1, Conv, [256, 3, 2]]                     #1x64x40x40
  - [[-1, 12], 1, Concat, [1]] # cat head P4        #1x192x40x40
  - [-1, 3, C2f, [512]] # 18 (P4/16-medium)       #1x128x40x40

  - [-1, 1, Conv, [512, 3, 2]]                     #1x128x20x20
  - [[-1, 9], 1, Concat, [1]] # cat head P5        #1x384x20x20
  - [-1, 3, C2f, [1024]] # 21 (P5/32-large)       #1x256x20x20

  - [[15, 18, 21], 1, Detect, [nc]] # Detect(P3, P4, P5)


