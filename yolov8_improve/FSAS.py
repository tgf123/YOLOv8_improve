import torch
import torch.nn as nn
import numbers
from einops import rearrange  # 用于方便地重排张量的库



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


# 将四维张量 (batch_size, channels, height, width) 转换为三维 (batch_size, height*width, channels)
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


# 将三维张量 (batch_size, height*width, channels) 转换回四维 (batch_size, channels, height, width)
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


# 带有 bias 的 LayerNorm 实现
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)  # 规范化形状参数

        assert len(normalized_shape) == 1  # 确保是1D输入

        # 创建可训练参数 weight 和 bias
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    # 前向传播
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)  # 计算每个样本的均值
        sigma = x.var(-1, keepdim=True, unbiased=False)  # 计算每个样本的方差
        # 进行归一化，并通过可训练的 weight 和 bias 调整
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


# 不带 bias 的 LayerNorm 实现
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1  # 确保是1D输入

        # 只有 weight 参数，没有 bias
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    # 前向传播
    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)  # 计算方差
        # 进行归一化，并通过可训练的 weight 调整
        return x / torch.sqrt(sigma + 1e-5) * self.weight


# LayerNorm 包装类，支持 BiasFree 和 WithBias 两种 LayerNorm 类型
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)  # 使用不带 bias 的 LayerNorm
        else:
            self.body = WithBias_LayerNorm(dim)  # 使用带 bias 的 LayerNorm

    # 前向传播，先将输入转换为 3D 进行 LayerNorm，再转换回原来的维度
    def forward(self, x):
        h, w = x.shape[-2:]  # 获取输入的高度和宽度
        return to_4d(self.body(to_3d(x)), h, w)


# FSAS 是核心模块，包含多个卷积操作和傅里叶变换
class FSAS(nn.Module):
    def __init__(self, dim, bias=False):
        super(FSAS, self).__init__()

        # 将输入通道数扩展到6倍 (for Q, K, V)
        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        # 使用深度可分离卷积对Q, K, V进行特征提取
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        # 输出的线性投影，最终将维度恢复到输入的dim
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        # LayerNorm，用于规范化 (这里是使用 WithBias 的版本)
        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

        self.patch_size = 4  # patch size，用于对 Q, K 做局部 FFT

    # 前向传播
    def forward(self, x):
        # 记录原始的高度和宽度
        original_h, original_w = x.shape[-2:]

        # 计算需要填充的高度和宽度
        pad_h = (self.patch_size - original_h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - original_w % self.patch_size) % self.patch_size

        # 对输入进行填充
        if pad_h > 0 or pad_w > 0:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h))

        # 使用 1x1 卷积扩展通道 (6倍扩展，用于 Q, K, V)
        hidden = self.to_hidden(x)

        # 使用深度卷积进一步提取 Q, K, V
        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)  # 按通道将 hidden 切成 Q, K, V

        # 对Q, K进行局部分块 (patch-wise)，方便后续进行局部FFT
        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)

        # 对Q, K进行2D FFT变换
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        # 进行点乘（傅里叶域中的乘法）
        out = q_fft * k_fft
        # 对结果进行逆傅里叶变换回到时域
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))

        # 恢复到原始分辨率
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        # 对结果进行 LayerNorm
        out = self.norm(out)

        # V 与 经过 FFT 后的 QK 相乘，产生最终输出
        output = v * out
        # 通过 1x1 卷积将维度还原回输入的维度
        output = self.project_out(output)

        # 裁剪回原始的高度和宽度
        if pad_h > 0 or pad_w > 0:
            output = output[:, :, :original_h, :original_w]

        return output


class Bottleneck_FSAS(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = FSAS(c_)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f_FSAS(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_FSAS(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

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



# 测试代码
if __name__ =='__main__':
    stars_Block =FSAS(256)  # 实例化 FSAS 模块，输入维度为256
    # 创建一个输入张量，形状为 (batch_size, C, H, W)
    batch_size = 8
    input_tensor = torch.randn(batch_size, 256, 21, 27)
    # 运行模型并打印输入和输出的形状
    output_tensor = stars_Block(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)


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
 #   - [-1, 6, C2f_FSAS, [256, True]]              # 1x64x80x80 -> 1x64x80x80
 #   - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16    1x64x80x80 -> 1x128x40x40
 #   - [-1, 6, C2f_FSAS, [512, True]]             #1x128x40x40-> 1x128x40x40
 #   - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32  1x128x40x40-> 1x256x20x20
 #   - [-1, 3, C2f_FSAS, [1024, True]]            # 1x256x20x20-> 1x256x20x20
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
