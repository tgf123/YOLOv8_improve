import torch
import torch.nn as nn
import torch.nn.functional as F

# https://arxiv.org/pdf/2209.14145
# Multi-scal eAttentio nNetwork for Single Image Super-Resolution

class LayerNorm(nn.Module):
    r"""支持两种数据格式的LayerNorm：channels_last（默认）或channels_first。
    输入的维度顺序。channels_last对应形状为(batch_size, height, width, channels)的输入，
    而channels_first对应形状为(batch_size, channels, height, width)的输入。
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        # 初始化参数，包括标准化参数weight和bias
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError  # 如果格式无效，则抛出异常
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        # 对于channels_last格式，使用F.layer_norm进行标准化
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # 对于channels_first格式，手动计算均值和方差进行标准化
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)  # 计算均值
            s = (x - u).pow(2).mean(1, keepdim=True)  # 计算方差
            x = (x - u) / torch.sqrt(s + self.eps)  # 标准化
            x = self.weight[:, None, None] * x + self.bias[:, None, None]  # 加上可学习的权重和偏置
            return x


class MLKA_Ablation(nn.Module):
    def __init__(self, n_feats, k=2, squeeze_factor=15):
        super().__init__()
        i_feats = 2 * n_feats  # 输入特征维度是n_feats的两倍

        self.n_feats = n_feats
        self.i_feats = i_feats

        # 初始化归一化层
        self.norm = LayerNorm(n_feats, data_format='channels_first')
        # 用于缩放的可学习参数
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        # 定义不同尺度的大核注意力（Large Kernel Attention）模块
        # LKA7: 使用7x7卷积，9x9卷积（膨胀为4），以及1x1卷积
        self.LKA7 = nn.Sequential(
            nn.Conv2d(n_feats // k, n_feats // k, 7, 1, 7 // 2, groups=n_feats // k),
            nn.Conv2d(n_feats // k, n_feats // k, 9, stride=1, padding=(9 // 2) * 4, groups=n_feats // k, dilation=4),
            nn.Conv2d(n_feats // k, n_feats // k, 1, 1, 0))

        # LKA5: 使用5x5卷积，7x7卷积（膨胀为3），以及1x1卷积
        self.LKA5 = nn.Sequential(
            nn.Conv2d(n_feats // k, n_feats // k, 5, 1, 5 // 2, groups=n_feats // k),
            nn.Conv2d(n_feats // k, n_feats // k, 7, stride=1, padding=(7 // 2) * 3, groups=n_feats // k, dilation=3),
            nn.Conv2d(n_feats // k, n_feats // k, 1, 1, 0))

        # 其它模块，未启用
        '''self.LKA3 = nn.Sequential(
            nn.Conv2d(n_feats//k, n_feats//k, 3, 1, 1, groups= n_feats//k),  
            nn.Conv2d(n_feats//k, n_feats//k, 5, stride=1, padding=(5//2)*2, groups=n_feats//k, dilation=2),
            nn.Conv2d(n_feats//k, n_feats//k, 1, 1, 0))'''

        # 定义其它卷积层（不同尺寸）
        self.X5 = nn.Conv2d(n_feats // k, n_feats // k, 5, 1, 5 // 2, groups=n_feats // k)
        self.X7 = nn.Conv2d(n_feats // k, n_feats // k, 7, 1, 7 // 2, groups=n_feats // k)

        # 用于将输入特征映射到更高维度的卷积层
        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, i_feats, 1, 1, 0))

        # 用于最终输出映射回n_feats维度的卷积层
        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))

    def forward(self, x, pre_attn=None, RAA=None):
        # 保留输入数据作为shortcut，进行跳跃连接
        shortcut = x.clone()

        # 对输入进行归一化
        x = self.norm(x)

        # 映射到更高维度
        x = self.proj_first(x)

        # 将特征图分成两部分（a和x）
        a, x = torch.chunk(x, 2, dim=1)

        # 将a进一步分成两部分（a_1和a_2）
        a_1, a_2 = torch.chunk(a, 2, dim=1)

        # 分别对a_1和a_2进行多尺度大核注意力操作，结合不同卷积结果
        a = torch.cat([self.LKA7(a_1) * self.X7(a_1), self.LKA5(a_2) * self.X5(a_2)], dim=1)

        # 最后一步将加权后的x和a做卷积变换，并进行跳跃连接
        x = self.proj_last(x * a) * self.scale + shortcut

        return x


if __name__ == '__main__':
    #############Test Model Complexity #############
    # from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis

    # 创建一个随机输入张量，大小为 (1, 256, 8, 8)
    x = torch.randn(1, 256, 8, 8)

    # 创建SAFMNPP模型
    model = MLKA_Ablation(256)
    print(model)

    # 测试模型的前向传播
    output = model(x)
    print(output.shape)  # 输出的形状



# elif m is MLKA_Ablation:
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
#   - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
#   - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
#   - [-1, 3, C2f, [128, True]]
#   - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
#   - [-1, 6, C2f, [256, True]]
#   - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
#   - [-1, 6, C2f, [512, True]]
#   - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
#   - [-1, 3, C2f, [1024, True]]
#   - [-1, 1, MLKA_Ablation, []] # 9
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
