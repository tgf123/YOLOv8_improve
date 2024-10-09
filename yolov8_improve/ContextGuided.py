###########################################################################
# Created by: Tianyi Wu
# Email: wutianyi@ict.ac.cn 
# Copyright (c) 2018
###########################################################################
import torch
import torch.nn as nn
from .conv import Conv

# Filter out variables, functions, and classes that other programs don't need or don't want when running cmd "from CGNet import *"



class BNPReLU(nn.Module):
    def __init__(self, nOut):
        """
        args:
           nOut: channels of output feature maps
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: normalized and thresholded feature map
        """
        output = self.bn(input)
        output = self.act(output)
        return output


class Conv1(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
            nIn: number of input channels
            nOut: number of output channels
            kSize: kernel size
            stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output


class ChannelWiseConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        Args:
            nIn: number of input channels
            nOut: number of output channels, default (nIn == nOut)
            kSize: kernel size
            stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn,
                              bias=False)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output


class DilatedConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels
           kSize: kernel size
           stride: optional stride rate for down-sampling
           d: dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False,
                              dilation=d)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output


class ChannelWiseDilatedConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels, default (nIn == nOut)
           kSize: kernel size
           stride: optional stride rate for down-sampling
           d: dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn,
                              bias=False, dilation=d)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output


class FGlo(nn.Module):
    """
    the FGlo class is employed to refine the joint feature of both local feature and surrounding context.
    """

    def __init__(self, channel, reduction=16):
        super(FGlo, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ContextGuidedBlock_Down(nn.Module):
    """
    the size of feature map divided 2, (H,W,C)---->(H/2, W/2, 2C)
    """

    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16):
        """
        args:
           nIn: the channel of input feature map
           nOut: the channel of output feature map, and nOut=2*nIn
        """
        super().__init__()
        self.conv1x1 = Conv(nIn, nOut, 3, 2)  # size/2, channel: nIn--->nOut

        self.F_loc = ChannelWiseConv(nOut, nOut, 3, 1)
        self.F_sur = ChannelWiseDilatedConv(nOut, nOut, 3, 1, dilation_rate)

        self.bn = nn.BatchNorm2d(2 * nOut, eps=1e-3)
        self.act = nn.PReLU(2 * nOut)
        self.reduce = Conv1(2 * nOut, nOut, 1, 1)  # reduce dimension: 2*nOut--->nOut

        self.F_glo = FGlo(nOut, reduction)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)

        joi_feat = torch.cat([loc, sur], 1)  # the joint feature
        joi_feat = self.bn(joi_feat)
        joi_feat = self.act(joi_feat)
        joi_feat = self.reduce(joi_feat)  # channel= nOut

        output = self.F_glo(joi_feat)  # F_glo is employed to refine the joint feature

        return output


class ContextGuidedBlock(nn.Module):
    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16, add=True):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels,
           add: if true, residual learning
        """
        super().__init__()
        n = int(nOut / 2)
        self.conv1x1 = Conv(nIn, n, 1, 1)  # 1x1 Conv is employed to reduce the computation
        self.F_loc = ChannelWiseConv(n, n, 3, 1)  # local feature
        self.F_sur = ChannelWiseDilatedConv(n, n, 3, 1, dilation_rate)  # surrounding context
        self.bn_prelu = BNPReLU(nOut)
        self.add = add
        self.F_glo = FGlo(nOut, reduction)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)

        joi_feat = torch.cat([loc, sur], 1)

        joi_feat = self.bn_prelu(joi_feat)

        output = self.F_glo(joi_feat)  # F_glo is employed to refine the joint feature
        # if residual version
        if self.add:
            output = input + output
        return output


class Bottleneck_CGuidedBlock(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ContextGuidedBlock(c1, c_)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f_CGuidedBlock(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_CGuidedBlock(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

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
    stars_Block =C2f_CGuidedBlock(256,256)
    # stars_Block = ContextGuidedBlock_Down(256, 256)
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
#   - [-1, 3, C2f, [128, True]]      #1x32x160x160 -> 1x32x160x160
#   - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8    1x32x160x160 -> 1x64x80x80
#   - [-1, 6, C2f, [256, True]]              # 1x64x80x80 -> 1x64x80x80
#   - [-1, 1, ContextGuidedBlock_Down, [512]] # 5-P4/16    1x64x80x80 -> 1x128x40x40
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
#
# from ultralytics.nn.modules.ContextGuided import ContextGuidedBlock_Down
#
# elif m is ContextGuidedBlock_Down:
# args = [ch[f], c2]


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
#   - [-1, 3, C2f_CGuidedBlock, [128, True]]      #1x32x160x160 -> 1x32x160x160
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

