import torch
from torch import nn
from torch.nn import functional as F


class NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None,  sub_sample=True, bn_layer=True):
        super(NonLocalBlockND, self).__init__()

        self.sub_sample = sub_sample  # 是否进行下采样

        self.in_channels = in_channels  # 输入通道数
        self.inter_channels = inter_channels  # 中间通道数

        # 如果未指定中间通道数，默认为输入通道数的一半
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1


        # 定义 g、theta、phi 的卷积层
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        # 定义 W 层，可选择是否使用批归一化
        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.in_channels)
            )
            # nn.init.constant(self.W[1].weight, 0)  # 初始化权重
            # nn.init.constant(self.W[1].bias, 0)    # 初始化偏置
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            # nn.init.constant(self.W.weight, 0)  # 初始化权重
            # nn.init.constant(self.W.bias, 0)    # 初始化偏置

        # 定义 theta 和 phi 的卷积层
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        # 如果进行下采样，则对 g 和 phi 添加池化层
        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, x):
        '''
        前向传播方法
        :param x: 输入张量，形状为 (b, c, t, h, w) （对于3D数据）
        :return: 输出张量，形状与输入相同
        '''
        batch_size = x.size(0)  # 获取批量大小

        # 计算 g(x)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # 变形为 (b, inter_channels, N)
        g_x = g_x.permute(0, 2, 1)  # 调整维度顺序

        # 计算 theta(x) 和 phi(x)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)  # 调整维度顺序
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        # 计算注意力权重
        f = torch.matmul(theta_x, phi_x)  # 矩阵乘法
        f_div_C = F.softmax(f, dim=-1)  # 归一化

        # 加权聚合
        y = torch.matmul(f_div_C, g_x)  # 使用权重对 g_x 加权
        y = y.permute(0, 2, 1).contiguous()  # 调整维度顺序
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  # 变形为 (b, inter_channels, t, h, w)

        # 融合输入和输出
        W_y = self.W(y)  # 通过 W 层
        z = W_y + x  # 残差连接

        return z  # 返回最终输出


if __name__ == '__main__':
    model = NonLocalBlockND(256)
    # 生成一个1x256x80x80的张量
    data = torch.randn(1, 256, 80, 80)
    updata = model.forward(data)
    print(updata.shape)  # 输出张量的形状



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
  - [-1, 3, NonLocalBlockND, [1024]]
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
  - [[-1, 13], 1, Concat, [1]] # cat head P4        #1x192x40x40
  - [-1, 3, C2f, [512]] # 18 (P4/16-medium)       #1x128x40x40

  - [-1, 1, Conv, [512, 3, 2]]                     #1x128x20x20
  - [[-1, 10], 1, Concat, [1]] # cat head P5        #1x384x20x20
  - [-1, 3, C2f, [1024]] # 21 (P5/32-large)       #1x256x20x20

  - [[16, 19, 22], 1, Detect, [nc]] # Detect(P3, P4, P5)

