# refer to the code from MogaNet, Thanks!
# https://github.com/Westlake-AI/MogaNet/blob/main/models/moganet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv



class ChannelAggregationFFN(nn.Module):
    """An implementation of FFN with Channel Aggregation in MogaNet."""

    def __init__(self, embed_dims, mlp_hidden_dims, kernel_size=3, act_layer=nn.GELU, ffn_drop=0.):
        super(ChannelAggregationFFN, self).__init__()
        self.embed_dims = embed_dims
        self.mlp_hidden_dims = mlp_hidden_dims

        self.fc1 = nn.Conv2d(
            in_channels=embed_dims, out_channels=self.mlp_hidden_dims, kernel_size=1)
        self.dwconv = nn.Conv2d(
            in_channels=self.mlp_hidden_dims, out_channels=self.mlp_hidden_dims, kernel_size=kernel_size,
            padding=kernel_size // 2, bias=True, groups=self.mlp_hidden_dims)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(
            in_channels=mlp_hidden_dims, out_channels=embed_dims, kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv2d(
            in_channels=self.mlp_hidden_dims, out_channels=1, kernel_size=1)
        self.sigma = nn.Parameter(
            1e-5 * torch.ones((1, mlp_hidden_dims, 1, 1)), requires_grad=True)
        self.decompose_act = act_layer()

    def feat_decompose(self, x):
        x = x + self.sigma * (x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        # proj 1
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        # proj 2
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiOrderDWConv(nn.Module):
    """Multi-order Features with Dilated DWConv Kernel in MogaNet."""

    def __init__(self, embed_dims, dw_dilation=[1, 2, 3], channel_split=[1, 3, 4]):
        super(MultiOrderDWConv, self).__init__()
        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims
        assert len(dw_dilation) == len(channel_split) == 3
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
        assert embed_dims % sum(channel_split) == 0

        # basic DW conv
        self.DW_conv0 = nn.Conv2d(
            in_channels=self.embed_dims, out_channels=self.embed_dims, kernel_size=5,
            padding=(1 + 4 * dw_dilation[0]) // 2,
            groups=self.embed_dims, stride=1, dilation=dw_dilation[0],
        )
        # DW conv 1
        self.DW_conv1 = nn.Conv2d(
            in_channels=self.embed_dims_1, out_channels=self.embed_dims_1, kernel_size=5,
            padding=(1 + 4 * dw_dilation[1]) // 2,
            groups=self.embed_dims_1, stride=1, dilation=dw_dilation[1],
        )
        # DW conv 2
        self.DW_conv2 = nn.Conv2d(
            in_channels=self.embed_dims_2, out_channels=self.embed_dims_2, kernel_size=7,
            padding=(1 + 6 * dw_dilation[2]) // 2,
            groups=self.embed_dims_2, stride=1, dilation=dw_dilation[2],
        )
        # a channel convolution
        self.PW_conv = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

    def forward(self, x):
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(
            x_0[:, self.embed_dims_0: self.embed_dims_0+self.embed_dims_1, ...])
        x_2 = self.DW_conv2(
            x_0[:, self.embed_dims-self.embed_dims_2:, ...])
        x = torch.cat([
            x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)
        x = self.PW_conv(x)
        return x


class MultiOrderGatedAggregation(nn.Module):
    """Spatial Block with Multi-order Gated Aggregation in MogaNet."""

    def __init__(self, embed_dims, attn_dw_dilation=[1, 2, 3], attn_channel_split=[1, 3, 4], attn_shortcut=True):
        super(MultiOrderGatedAggregation, self).__init__()
        self.embed_dims = embed_dims
        self.attn_shortcut = attn_shortcut
        self.proj_1 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.gate = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.value = MultiOrderDWConv(
            embed_dims=embed_dims, dw_dilation=attn_dw_dilation, channel_split=attn_channel_split)
        self.proj_2 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

        # activation for gating and value
        self.act_value = nn.SiLU()
        self.act_gate = nn.SiLU()
        # decompose
        self.sigma = nn.Parameter(1e-5 * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)

    def feat_decompose(self, x):
        x = self.proj_1(x)
        # x_d: [B, C, H, W] -> [B, C, 1, 1]
        x_d = F.adaptive_avg_pool2d(x, output_size=1)
        x = x + self.sigma * (x - x_d)
        x = self.act_value(x)
        return x

    def forward(self, x):
        if self.attn_shortcut:
            shortcut = x.clone()
        # proj 1x1
        x = self.feat_decompose(x)
        # gating and value branch
        g = self.gate(x)
        v = self.value(x)
        # aggregation
        x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        if self.attn_shortcut:
            x = x + shortcut
        return x

class Bottleneck_MultiOGA(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = MultiOrderGatedAggregation(c1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f_MultiOGA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_MultiOGA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

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


第一种改进
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
  - [-1, 3, C2f_MultiOGA, [128, True]]             #1x32x160x160 -> 1x32x160x160
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

if m in {BottleneckCSP, C1, C2, C2f, C2fAttn, C3, C3TR, C3Ghost, C3x, RepC3, C2f_MultiOGA}:


第二种改进

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
  - [-1, 3, C2f, [128, True]]             #1x32x160x160 -> 1x32x160x160
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8    1x32x160x160 -> 1x64x80x80
  - [-1, 6, C2f, [256, True]]              # 1x64x80x80 -> 1x64x80x80
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16    1x64x80x80 -> 1x128x40x40
  - [-1, 6, C2f, [512, True]]             #1x128x40x40-> 1x128x40x40
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32  1x128x40x40-> 1x256x20x20
  - [-1, 3, C2f, [1024, True]]            # 1x256x20x20-> 1x256x20x20
  - [-1, 1, SPPF, [1024, 5]] # 9             1x256x20x20-> 1x256x20x20
  - [-1, 1, ChannelAggregationFFN, [512]]

# YOLOv8.0n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 1x256x40x40
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4  # # 1x384x40x40
  - [-1, 3, C2f, [512]] # 12                       1x128x40x40

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] #   1x128x80x80
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3    1x192x80x80
  - [-1, 3, C2f, [256]] # 15 (P3/8-small)          1x64x80x80

  - [-1, 1, Conv, [256, 3, 2]]                     #1x64x40x40
  - [[-1, 13], 1, Concat, [1]] # cat head P4        #1x192x40x40
  - [-1, 3, C2f, [512]] # 18 (P4/16-medium)       #1x128x40x40

  - [-1, 1, Conv, [512, 3, 2]]                     #1x128x20x20
  - [[-1, 9], 1, Concat, [1]] # cat head P5        #1x384x20x20
  - [-1, 3, C2f, [1024]] # 21 (P5/32-large)       #1x256x20x20

  - [[16, 19, 22], 1, Detect, [nc]] # Detect(P3, P4, P5)

elif m is ChannelAggregationFFN:
args = [ch[f], c2]

第三种改进

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
  - [-1, 3, C2f, [128, True]]             #1x32x160x160 -> 1x32x160x160
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8    1x32x160x160 -> 1x64x80x80
  - [-1, 6, C2f, [256, True]]              # 1x64x80x80 -> 1x64x80x80
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16    1x64x80x80 -> 1x128x40x40
  - [-1, 6, C2f, [512, True]]             #1x128x40x40-> 1x128x40x40
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32  1x128x40x40-> 1x256x20x20
  - [-1, 3, C2f, [1024, True]]            # 1x256x20x20-> 1x256x20x20
  - [-1, 1, SPPF, [1024, 5]] # 9             1x256x20x20-> 1x256x20x20
  - [-1, 1, MultiOrderGatedAggregation, [1024]]

# YOLOv8.0n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 1x256x40x40
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4  # # 1x384x40x40
  - [-1, 3, C2f, [512]] # 12                       1x128x40x40

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] #   1x128x80x80
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3    1x192x80x80
  - [-1, 3, C2f, [256]] # 15 (P3/8-small)          1x64x80x80

  - [-1, 1, Conv, [256, 3, 2]]                     #1x64x40x40
  - [[-1, 13], 1, Concat, [1]] # cat head P4        #1x192x40x40
  - [-1, 3, C2f, [512]] # 18 (P4/16-medium)       #1x128x40x40

  - [-1, 1, Conv, [512, 3, 2]]                     #1x128x20x20
  - [[-1, 9], 1, Concat, [1]] # cat head P5        #1x384x20x20
  - [-1, 3, C2f, [1024]] # 21 (P5/32-large)       #1x256x20x20

  - [[16, 19, 22], 1, Detect, [nc]] # Detect(P3, P4, P5)

elif m is MultiOrderGatedAggregation:
args = [ch[f]]