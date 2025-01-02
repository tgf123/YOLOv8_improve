import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from thop import clever_format, profile
# from torchsummary import summary


# https://link.springer.com/chapter/10.1007/978-3-031-72114-4_60
# TinyU-Net: Lighter Yet Better U-Net with Cascaded Multi-receptive Fields



def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))



class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


# Lightweight Cascade Multi-Receptive Fields Module
class CMRF(nn.Module):
    """CMRF Module with args(ch_in, ch_out, number, shortcut, groups, expansion)."""

    def __init__(self, c1, c2, N=8, shortcut=True, g=1, e=0.5):
        super().__init__()

        self.N = N
        self.c = int(c2 * e / self.N)
        self.add = shortcut and c1 == c2

        self.pwconv1 = Conv(c1, c2 // self.N, 1, 1)
        self.pwconv2 = Conv(c2 // 2, c2, 1, 1)
        self.m = nn.ModuleList(DWConv(self.c, self.c, k=3, act=False) for _ in range(N - 1))

    def forward(self, x):
        """Forward pass through CMRF Module."""
        x_residual = x
        x = self.pwconv1(x)

        x = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
        x.extend(m(x[-1]) for m in self.m)
        x[0] = x[0] + x[1]
        x.pop(1)

        y = torch.cat(x, dim=1)
        y = self.pwconv2(y)
        return x_residual + y if self.add else y




if __name__ == '__main__':


    # 定义输入张量的形状为 [16, 512, 32, 32]
    input_tensor = torch.randn(1, 128, 32, 32)

    # 创建 ConvolutionalGLU 模块
    conv_glu = CMRF(128, 256)

    # 输入特征图的高度和宽度
    H = 32
    W = 32

    # 前向传播
    output = conv_glu(input_tensor)
    print("Output shape:", output.shape)



# # Ultralytics YOLO 🚀, AGPL-3.0 license
# # YOLOv8 object detection model with P3-P5 outputs. For Usage examples see https://docs.ultralytics.com/tasks/detect
#
# # Parameters
# nc: 80 # number of classes
# scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
#   # [depth, width, max_channels]
#   n: [0.33, 0.25, 1024] # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
#   s: [0.33, 0.50, 1024] # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
#   m: [0.67, 0.75, 768] # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
#   l: [1.00, 1.00, 512] # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
#   x: [1.00, 1.25, 512] # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs
#
# # YOLOv8.0n backbone
# backbone:
#   # [from, repeats, module, args]
#   - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
#   - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
#   - [-1, 3, C2f, [128, True]]
#   - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
#   - [-1, 6, C2f, [256, True]]
#   - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
#   - [-1, 6, C2f, [512, True]]
#   - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
#   - [-1, 3, C2f, [1024, True]]
#   - [-1, 1, CMRF, [1024]] # 9
#
# # YOLOv8.0n head
# head:
#   - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
#   - [[-1, 6], 1, Concat, [1]] # cat backbone P4
#   - [-1, 3, C2f, [512]] # 12
#
#   - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
#   - [[-1, 4], 1, Concat, [1]] # cat backbone P3
#   - [-1, 3, C2f, [256]] # 15 (P3/8-small)
#
#   - [-1, 1, Conv, [256, 3, 2]]
#   - [[-1, 12], 1, Concat, [1]] # cat head P4
#   - [-1, 3, C2f, [512]] # 18 (P4/16-medium)
#
#   - [-1, 1, Conv, [512, 3, 2]]
#   - [[-1, 9], 1, Concat, [1]] # cat head P5
#   - [-1, 3, C2f, [1024]] # 21 (P5/32-large)
#
#   - [[15, 18, 21], 1, Detect, [nc]] # Detect(P3, P4, P5)

