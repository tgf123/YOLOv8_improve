"""
Implementation of Prof-of-Concept Network: StarNet.

We make StarNet as simple as possible [to show the key contribution of element-wise multiplication]:
    - like NO layer-scale in network design,
    - and NO EMA during training,
    - which would improve the performance further.

Created by: Xu Ma (Email: ma.xu1@northeastern.edu)
Modified Date: Mar/29/2024
"""
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from .conv import Conv


class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class StarsBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x

class Bottleneck_StarsBlock(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = StarsBlock(c1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f_StarsBlock(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_StarsBlock(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

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
    stars_Block =StarsBlock(dim=3)
    #创建一个输入张量，形状为(batch_size, H*W,C)
    batch_size = 8
    H, W = (32, 32)
    input_tensor=torch.randn(batch_size,3, 64,64 )
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
#   - [-1, 3, C2f, [128, True]]      #1x32x160x160 -> 1x32x160x160
#   - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8    1x32x160x160 -> 1x64x80x80
#   - [-1, 6, C2f, [256, True]]              # 1x64x80x80 -> 1x64x80x80
#   - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16    1x64x80x80 -> 1x128x40x40
#   - [-1, 6, C2f, [512, True]]             #1x128x40x40-> 1x128x40x40
#   - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32  1x128x40x40-> 1x256x20x20
#   - [-1, 3, C2f, [1024, True]]            # 1x256x20x20-> 1x256x20x20
#   - [-1, 1, StarsBlock, [1024]]
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
#   - [[-1, 13], 1, Concat, [1]] # cat head P4        #1x192x40x40
#   - [-1, 3, C2f, [512]] # 18 (P4/16-medium)       #1x128x40x40
#
#   - [-1, 1, Conv, [512, 3, 2]]                     #1x128x20x20
#   - [[-1, 10], 1, Concat, [1]] # cat head P5        #1x384x20x20
#   - [-1, 3, C2f, [1024]] # 21 (P5/32-large)       #1x256x20x20
#
#   - [[16, 19, 22], 1, Detect, [nc]] # Detect(P3, P4, P5)

# elif m is StarsBlock:
#     args = [ch[f]]


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
#   - [-1, 3, C2f_StarsBlock, [128, True]]      #1x32x160x160 -> 1x32x160x160
#   - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8    1x32x160x160 -> 1x64x80x80
#   - [-1, 6, C2f, [256, True]]              # 1x64x80x80 -> 1x64x80x80
#   - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16    1x64x80x80 -> 1x128x40x40
#   - [-1, 6, C2f, [512, True]]             #1x128x40x40-> 1x128x40x40
#   - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32  1x128x40x40-> 1x256x20x20
#   - [-1, 3, C2f, [1024, True]]            # 1x256x20x20-> 1x256x20x20
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
