import torch
import torch.nn as nn



class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule, self).__init__()
        # 平均池化和最大池化操作，用于生成不同的特征表示
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 两层卷积网络用于生成通道注意力权重
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        # 使用 Sigmoid 激活函数将权重限制在 [0, 1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 计算平均池化和最大池化特征
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))

        # 将两个池化特征相加
        out = avg_out + max_out
        return self.sigmoid(out)  # 返回通道注意力权重


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()

        # 使用 7x7 卷积实现空间注意力
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 计算每个通道的平均和最大特征
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # 将两个特征在通道维度上合并
        x = torch.cat([avg_out, max_out], dim=1)

        # 使用 7x7 卷积生成空间注意力图
        x = self.conv1(x)
        return self.sigmoid(x)  # 返回空间注意力权重


class MSAA(nn.Module):
    def __init__(self, in_channels, out_channels, factor=4.0):
        super(MSAA, self).__init__()

        # 下采样卷积将输入通道减少到 dim
        dim = int(out_channels // factor)
        self.down = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)

        # 3x3, 5x5 和 7x7 不同大小卷积核提取多尺度特征
        self.conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
        self.conv_7x7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3)

        # 引入空间和通道注意力模块
        self.spatial_attention = SpatialAttentionModule()
        self.channel_attention = ChannelAttentionModule(dim)

        # 上采样卷积将通道数恢复到原始的 out_channels
        self.up = nn.Conv2d(dim, out_channels, kernel_size=1, stride=1)

        # 再次降维的卷积，便于多尺度特征融合
        self.down_2 = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)

    def forward(self, x):
        # 将输入降维，提取多尺度特征
        x = self.down(x)

        # 应用通道注意力机制
        x = x * self.channel_attention(x)

        # 使用多尺度卷积提取特征
        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        x_7x7 = self.conv_7x7(x)

        # 将多尺度特征相加并应用空间注意力机制
        x_s = x_3x3 + x_5x5 + x_7x7
        x_s = x_s * self.spatial_attention(x_s)

        # 将处理后的特征与原始特征相加，并恢复到原始的通道数
        x_out = self.up(x_s + x)

        return x_out  # 返回输出特征


if __name__ =='__main__':
    MSAA =MSAA(256,256)
    #创建一个输入张量，形状为(batch_size, H*W,C)
    batch_size = 8
    input_tensor=torch.randn(batch_size, 256, 64, 64 )
    #运行模型并打印输入和输出的形状
    output_tensor =MSAA(input_tensor)
    print("Input shape:",input_tensor.shape)
    print("0utput shape:",output_tensor.shape)

# elif m is MSAA:
#     args = [ch[f], ch[f]]


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
#   - [-1, 1, SPPF, [1024, 5]] # 9             1x256x20x20-> 1x256x20x20
#
# # YOLOv8.0n head
# head:
#   - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 1x256x40x40
#   - [[-1, 6], 1, Concat, [1]] # cat backbone P4  # # 1x384x40x40
#   - [-1, 1, MSAA, [0]]
#   - [-1, 3, C2f, [512]] # 13                       1x128x40x40
#
#   - [-1, 1, nn.Upsample, [None, 2, "nearest"]] #   1x128x80x80
#   - [[-1, 4], 1, Concat, [1]] # cat backbone P3    1x192x80x80
#   - [-1, 1, MSAA, [0]]
#   - [-1, 3, C2f, [256]] # 17 (P3/8-small)          1x64x80x80
#
#   - [-1, 1, Conv, [256, 3, 2]]                     #1x64x40x40
#   - [[-1, 13], 1, Concat, [1]] # cat head P4        #1x192x40x40
#   - [-1, 1, MSAA, [0]]
#   - [-1, 3, C2f, [512]] # 21 (P4/16-medium)       #1x128x40x40
#
#   - [-1, 1, Conv, [512, 3, 2]]                     #1x128x20x20
#   - [[-1, 9], 1, Concat, [1]] # cat head P5        #1x384x20x20
#   - [-1, 1, MSAA, [0]]
#   - [-1, 3, C2f, [1024]] # 25 (P5/32-large)       #1x256x20x20
#
#   - [[17, 21, 25], 1, Detect, [nc]] # Detect(P3, P4, P5)
