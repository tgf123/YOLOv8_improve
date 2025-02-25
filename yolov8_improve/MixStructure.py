import torch
import torch.nn as nn


class MixStructureBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # 批量归一化层 (用于规范化激活值)
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)

        # 卷积层 (1x1, 5x5, 7x7, 5x5, 3x3)
        # 这些卷积层用于提取不同感受野的特征
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)  # 1x1卷积，将输入映射到相同维度
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, padding_mode='reflect')  # 5x5卷积，采用反射填充

        # 3个不同的卷积层，具有不同的卷积核大小和扩张操作
        self.conv3_19 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, padding_mode='reflect')  # 7x7卷积，带有深度可分离卷积
        self.conv3_13 = nn.Conv2d(dim, dim, kernel_size=5, padding=6, groups=dim, dilation=3, padding_mode='reflect')  # 5x5卷积，带扩张操作
        self.conv3_7 = nn.Conv2d(dim, dim, kernel_size=3, padding=3, groups=dim, dilation=3, padding_mode='reflect')  # 3x3卷积，带扩张操作

        # 简单像素级注意力
        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),  # 1x1卷积
            nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim, padding_mode='reflect')  # 3x3卷积
        )
        self.Wg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 自适应平均池化
            nn.Conv2d(dim, dim, 1),  # 1x1卷积
            nn.Sigmoid()  # Sigmoid激活
        )

        # 通道注意力机制
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 自适应平均池化
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),  # 1x1卷积
            nn.GELU(),  # GELU激活函数
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),  # 1x1卷积
            nn.Sigmoid()  # Sigmoid激活
        )

        # 像素级注意力机制
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),  # 1x1卷积
            nn.GELU(),  # GELU激活函数
            nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True),  # 1x1卷积
            nn.Sigmoid()  # Sigmoid激活
        )

        # MLP层，用于将不同的卷积结果进行融合
        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),  # 1x1卷积，通道数扩展
            nn.GELU(),  # GELU激活
            nn.Conv2d(dim * 4, dim, 1)  # 1x1卷积，恢复通道数
        )

        # 另一个MLP层，和上面类似，但输入不同
        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),  # 1x1卷积
            nn.GELU(),  # GELU激活
            nn.Conv2d(dim * 4, dim, 1)  # 1x1卷积
        )

    def forward(self, x):
        identity = x  # 保存输入，以便后续进行残差连接
        
        # 第一个卷积块
        x = self.norm1(x)  # 批量归一化
        x = self.conv1(x)  # 1x1卷积
        x = self.conv2(x)  # 5x5卷积
        # 将3种不同的卷积结果合并在一起
        x = torch.cat([self.conv3_19(x), self.conv3_13(x), self.conv3_7(x)], dim=1)
        x = self.mlp(x)  # 通过MLP进行融合
        x = identity + x  # 残差连接，增强特征

        # 第二个卷积块
        identity = x  # 保存输入，以便后续进行残差连接
        x = self.norm2(x)  # 批量归一化
        # 进行注意力机制操作
        x = torch.cat([self.Wv(x) * self.Wg(x), self.ca(x) * x, self.pa(x) * x], dim=1)
        x = self.mlp2(x)  # 通过MLP进行融合
        x = identity + x  # 残差连接，增强特征

        return x  # 返回最终输出


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


class Bottleneck_MixStructureBlock(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.cv3 = MixStructureBlock(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))


class C2f_MixStructure(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_MixStructureBlock(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

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

def main():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)

    # 定义输入张量的形状 (batch_size, channels, height, width)

    # 创建一个随机输入张量
    x = torch.randn(4, 64, 27, 32)

    # 打印输入张量的形状
    print(f"Input shape: {x.shape}")

    # 初始化 MixStructureBlock 模块
    mix_structure_block = MixStructureBlock(dim=64)

    # 前向传播
    output = mix_structure_block(x)

    # 打印输出张量的形状
    print(f"Output shape: {output.shape}")

    # 检查输入和输出形状是否一致
    assert x.shape == output.shape, "Input and output shapes do not match!"

    print("MixStructureBlock forward pass successful!")

if __name__ == "__main__":
    main()



# # Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# # Ultralytics YOLOv8 object detection model with P3/8 - P5/32 outputs
# # Model docs: https://docs.ultralytics.com/models/yolov8
# # Task docs: https://docs.ultralytics.com/tasks/detect
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
#   - [-1, 3, C2f_MixStructure, [128, True]]
#   - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
#   - [-1, 6, C2f_MixStructure, [256, True]]
#   - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
#   - [-1, 6, C2f_MixStructure, [512, True]]
#   - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
#   - [-1, 3, C2f_MixStructure, [1024, True]]
#   - [-1, 1, SPPF, [1024, 5]] # 9
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







