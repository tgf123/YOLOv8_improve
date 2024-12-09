import torch
import torch.nn as nn
import torch.nn.functional as F


# 论文地址：https://openaccess.thecvf.com/content/ICCV2023/papers/Sun_Spatially-Adaptive_Feature_Modulation_for_Efficient_Image_Super-Resolution_ICCV_2023_paper.pdf
# 论文题目：Spatially-Adaptive Feature Modulation for Efficient Image Super-Resolution
# 该论文提出了一种空间自适应特征调制（SAFM）方法来有效地进行图像超分辨率

class SimpleSAFM(nn.Module):
    def __init__(self, dim, ratio=4):
        """
        初始化SimpleSAFM模块。
        dim: 输入的特征图通道数。
        ratio: 将通道数分为两部分的比例。
        """
        super().__init__()
        self.dim = dim
        self.chunk_dim = dim // ratio  # 将输入通道数划分为两部分

        # 卷积层，用于特征映射的投影
        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)

        # 深度卷积，用于处理分割后的特征
        self.dwconv = nn.Conv2d(self.chunk_dim, self.chunk_dim, 3, 1, 1, groups=self.chunk_dim, bias=False)

        # 输出卷积层
        self.out = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)

        # 激活函数
        self.act = nn.GELU()

    def forward(self, x):
        """
        前向传播：进行特征提取与调制
        x: 输入特征图，大小为 (batch_size, dim, H, W)
        """
        h, w = x.size()[-2:]  # 获取输入的高度和宽度

        # 将输入分为两部分，x0 和 x1
        x0, x1 = self.proj(x).split([self.chunk_dim, self.dim - self.chunk_dim], dim=1)

        # 对x0进行池化，并通过深度卷积进行处理
        x2 = F.adaptive_max_pool2d(x0, (h // 8, w // 8))  # 自适应池化，缩小为原图的1/8大小
        x2 = self.dwconv(x2)  # 深度卷积
        x2 = F.interpolate(x2, size=(h, w), mode='bilinear')  # 上采样恢复到原尺寸
        x2 = self.act(x2) * x0  # 激活并与x0相乘，进行特征调制

        # 将x1和调制后的x2合并，进行后续处理
        x = torch.cat([x1, x2], dim=1)
        x = self.out(self.act(x))  # 输出
        return x


# Convolutional Channel Mixer 模块
class CCM(nn.Module):
    def __init__(self, dim, ffn_scale, use_se=False):
        """
        初始化CCM模块。
        dim: 输入的特征图通道数。
        ffn_scale: 用于计算隐藏层维度的比例。
        use_se: 是否使用Squeeze-and-Excitation（SE）模块
        """
        super().__init__()
        self.use_se = use_se
        hidden_dim = int(dim * ffn_scale)  # 隐藏层通道数

        # 卷积层1：进行通道扩展
        self.conv1 = nn.Conv2d(dim, hidden_dim, 3, 1, 1, bias=False)

        # 卷积层2：将通道恢复到原始维度
        self.conv2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0, bias=False)

        self.act = nn.GELU()  # 激活函数

    def forward(self, x):
        """
        前向传播：进行特征混合
        x: 输入特征图，大小为 (batch_size, dim, H, W)
        """
        x = self.act(self.conv1(x))  # 经过卷积和激活
        x = self.conv2(x)  # 经过第二层卷积
        return x


# Attention Block 模块
class AttBlock(nn.Module):
    def __init__(self, dim, ffn_scale, use_se=False):
        """
        初始化Attention Block模块，该模块包含SimpleSAFM和CCM
        """
        super().__init__()

        # SimpleSAFM：空间自适应特征调制
        self.conv1 = SimpleSAFM(dim, ratio=3)

        # CCM：卷积通道混合
        self.conv2 = CCM(dim, ffn_scale, use_se)

    def forward(self, x):
        """
        前向传播：依次经过SimpleSAFM和CCM模块
        x: 输入特征图，大小为 (batch_size, dim, H, W)
        """
        out = self.conv1(x)  # 经过SimpleSAFM模块
        out = self.conv2(out)  # 经过CCM模块
        return out + x  # 残差连接


# SAFM与NPP（Non-Local Prior Pooling）结合的超分辨率网络
class SAFMNPP(nn.Module):
    def __init__(self, input_dim, dim, n_blocks=3, ffn_scale=1.5, use_se=False, upscaling_factor=2):
        """
        初始化SAFMNPP超分辨率模型。
        input_dim: 输入图像的通道数。
        dim: 网络中间层的通道数。
        n_blocks: 注意力块的数量。
        ffn_scale: 用于计算通道数的比例。
        use_se: 是否使用Squeeze-and-Excitation（SE）模块。
        upscaling_factor: 超分辨率放大因子。
        """
        super().__init__()
        self.scale = upscaling_factor  # 设置上采样因子

        # 输入特征图转换
        self.to_feat = nn.Conv2d(input_dim, dim, 3, 1, 1, bias=False)

        # 堆叠多个Attention块
        self.feats = nn.Sequential(*[AttBlock(dim, ffn_scale, use_se) for _ in range(n_blocks)])

        # 输出恢复图像的卷积层
        self.to_img = nn.Sequential(
            nn.Conv2d(dim, input_dim * upscaling_factor ** 2, 3, 1, 1, bias=False),
            nn.PixelShuffle(upscaling_factor)  # 用PixelShuffle进行上采样
        )

    def forward(self, x):
        """
        前向传播：输入图像经过多个模块进行超分辨率恢复
        x: 输入图像，大小为 (batch_size, input_dim, H, W)
        """
        # 首先通过插值进行上采样
        res = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)

        # 通过to_feat模块将输入特征转换到中间层
        x = self.to_feat(x)

        # 通过多个Attention块进行特征处理
        x = self.feats(x)

        # 通过to_img模块进行恢复图像
        return self.to_img(x) + res  # 最终输出图像加上残差


# 测试模型的复杂度
if __name__ == '__main__':
    #############Test Model Complexity #############
    # from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis

    # 创建一个随机输入张量，大小为 (1, 256, 8, 8)
    x = torch.randn(1, 256, 8, 8)

    # 创建SAFMNPP模型
    model = SAFMNPP(256, dim=256, n_blocks=6, ffn_scale=1.5, upscaling_factor=2)
    print(model)

    # 测试模型的前向传播
    output = model(x)
    print(output.shape)  # 输出的形状

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
#   - [-1, 1, SPPF, [1024, 5]] # 9
#
# # YOLOv8.0n head
# head:
#   - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
#   - [[-1, 6], 1, Concat, [1]] # cat backbone P4
#   - [-1, 3, C2f, [512]] # 12
#
#   - [-1, 1, SAFMNPP, [512]]
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
