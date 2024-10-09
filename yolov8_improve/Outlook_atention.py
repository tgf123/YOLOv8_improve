import torch
import torch.nn as nn
import torch.nn.functional as F
# from conv import Conv
from .conv import Conv

import math


class OutlookAttention(nn.Module):
    """
    Implementation of outlook attention
    --dim: hidden dim
    --num_heads: number of heads
    --kernel_size: kernel size in each window for outlook attention
    return: token features after outlook attention
    """

    def __init__(self, dim, num_heads=1, kernel_size=3, padding=1, stride=1,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        # 计算每个注意力头的维度
        head_dim = dim // num_heads
        self.num_heads = num_heads  # 注意力头的数量
        self.kernel_size = kernel_size  # 卷积核大小
        self.padding = padding  # 卷积填充
        self.stride = stride  # 卷积步幅

        # QK 的缩放因子，默认是头维度的倒数平方根
        self.scale = qk_scale or head_dim ** -0.5

        # 定义线性层，用于计算值 V
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        # 定义线性层，用于计算注意力权重
        self.attn = nn.Linear(dim, kernel_size ** 4 * num_heads)

        # 定义丢弃层，用于注意力计算的丢弃
        self.attn_drop = nn.Dropout(attn_drop)
        # 定义输出投影层
        self.proj = nn.Linear(dim, dim)
        # 定义输出的丢弃层
        self.proj_drop = nn.Dropout(proj_drop)

        # 定义展开操作，将输入特征图转化为局部窗口
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        # 定义平均池化操作，用于生成上下文信息
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

    def forward(self, x):
        # 调整输入维度，从 (B, C, H, W) 转为 (B, H, W, C)
        x = x.permute(0, 2, 3, 1)

        B, H, W, C = x.shape  # 解包输入特征图的维度

        # 计算值 V，并调整维度为 (B, C, H, W)
        v = self.v(x).permute(0, 3, 1, 2)

        # 计算经过步幅处理后的特征图高度和宽度
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)

        # 将值 V 展开为局部窗口，调整形状为 (B, H, N, kxk, C/H)
        v = self.unfold(v).reshape(B, self.num_heads, C // self.num_heads,
                                   self.kernel_size * self.kernel_size,
                                   h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H

        # 对输入特征图进行平均池化，生成上下文信息
        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # 计算注意力权重并调整形状为 (B, H, N, kxk, kxk)
        attn = self.attn(attn).reshape(
            B, h * w, self.num_heads, self.kernel_size * self.kernel_size,
               self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)

        # 缩放注意力权重并进行 softmax 归一化
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # 应用丢弃

        # 使用注意力权重对值 V 进行加权求和
        x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
            B, C * self.kernel_size * self.kernel_size, h * w)

        # 将特征图重构为原始尺寸
        x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size,
                   padding=self.padding, stride=self.stride)

        # 通过线性层进行输出投影
        x = self.proj(x.permute(0, 2, 3, 1))
        x = self.proj_drop(x)  # 应用丢弃

        # 将输出维度调整回 (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return x  # 返回处理后的特征图


class Bottleneck_OAtention(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.cv3 = OutlookAttention( c2, 4)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))


class C2f_OAtention(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_OAtention(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

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



if __name__ =='__main__':
    stars_Block =OutlookAttention(256)
    #创建一个输入张量，形状为(batch_size, H*W,C)
    batch_size = 8
    input_tensor=torch.randn(batch_size, 256, 64, 64 )
    #运行模型并打印输入和输出的形状
    output_tensor =stars_Block(input_tensor)
    print("Input shape:",input_tensor.shape)
    print("0utput shape:",output_tensor.shape)


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
#   - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2     1x3x640x640 -> 1x16x320x320
#   - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4   1x16x320x320 -> 1x32x160x160
#   - [-1, 3, C2f_OAtention, [128, True]]      #1x32x160x160 -> 1x32x160x160
#   - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8    1x32x160x160 -> 1x64x80x80
#   - [-1, 6, C2f_OAtention, [256, True]]              # 1x64x80x80 -> 1x64x80x80
#   - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16    1x64x80x80 -> 1x128x40x40
#   - [-1, 6, C2f_OAtention, [512, True]]             #1x128x40x40-> 1x128x40x40
#   - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32  1x128x40x40-> 1x256x20x20
#   - [-1, 3, C2f_OAtention, [1024, True]]            # 1x256x20x20-> 1x256x20x20
#   - [-1, 1, SPPF, [1024, 5]] # 9             1x256x20x20-> 1x256x20x20
#
# # YOLOv8.0n head
# head:
#   - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 1x256x40x40
#   - [[-1, 6], 1, Concat, [1]] # cat backbone P4  # # 1x384x40x40
#   - [-1, 3, C2f, [512]] # 12                       1x128x40x40
#
#   - [-1, 1, nn.Upsample, [None, 2, "nearest"]] #   1x128x80x80
#   - [[-1, 4], 1, Concat, [1]] # cat backbone P3    1x192x80x80
#   - [-1, 3, C2f, [256]] # 15 (P3/8-small)          1x64x80x80
#
#   - [-1, 1, Conv, [256, 3, 2]]                     #1x64x40x40
#   - [[-1, 12], 1, Concat, [1]] # cat head P4        #1x192x40x40
#   - [-1, 3, C2f, [512]] # 18 (P4/16-medium)       #1x128x40x40
#
#   - [-1, 1, Conv, [512, 3, 2]]                     #1x128x20x20
#   - [[-1, 9], 1, Concat, [1]] # cat head P5        #1x384x20x20
#   - [-1, 3, C2f, [1024]] # 21 (P5/32-large)       #1x256x20x20
#
#   - [[15, 18, 21], 1, Detect, [nc]] # Detect(P3, P4, P5)
