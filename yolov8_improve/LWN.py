import torch
from torch import nn
import pywt
from typing import Sequence, Tuple, Union, List
from einops import rearrange, repeat
import torch.nn.functional as F


# 用于获取滤波器张量的函数
def get_filter_tensors(
        wavelet,
        flip: bool,
        device: Union[torch.device, str] = 'cpu',
        dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """将输入的wavelet转换为滤波器张量。

    参数：
        wavelet (Wavelet 或 str): 一个pywt wavelet兼容的对象，或者是一个pywt wavelet的名称。
        flip (bool): 如果为True，滤波器将被翻转。
        device (torch.device): PyTorch目标设备。默认值为'cpu'。
        dtype (torch.dtype): 数据类型，设置计算的精度。默认是torch.float32。

    返回：
        tuple: 返回包含四个滤波器张量的元组（dec_lo, dec_hi, rec_lo, rec_hi）。
    """
    wavelet = _as_wavelet(wavelet)

    # 用于创建滤波器张量的辅助函数
    def _create_tensor(filter: Sequence[float]) -> torch.Tensor:
        if flip:
            # 如果需要翻转滤波器
            if isinstance(filter, torch.Tensor):
                return filter.flip(-1).unsqueeze(0).to(device)
            else:
                return torch.tensor(filter[::-1], device=device, dtype=dtype).unsqueeze(0)
        else:
            # 不翻转滤波器
            if isinstance(filter, torch.Tensor):
                return filter.unsqueeze(0).to(device)
            else:
                return torch.tensor(filter, device=device, dtype=dtype).unsqueeze(0)

    # 获取小波滤波器
    dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
    dec_lo_tensor = _create_tensor(dec_lo)
    dec_hi_tensor = _create_tensor(dec_hi)
    rec_lo_tensor = _create_tensor(rec_lo)
    rec_hi_tensor = _create_tensor(rec_hi)
    return dec_lo_tensor, dec_hi_tensor, rec_lo_tensor, rec_hi_tensor


# 将波形转换为pywt wavelet对象的辅助函数
def _as_wavelet(wavelet):
    if isinstance(wavelet, str):
        return pywt.Wavelet(wavelet)
    else:
        return wavelet


# ShuffleBlock模块，用于通道重排
class ShuffleBlock(nn.Module):
    def __init__(self, groups=2):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        # 重排张量的维度
        x = rearrange(x, 'b (g f) h w -> b g f h w', g=self.groups)
        x = rearrange(x, 'b g f h w -> b f g h w')
        x = rearrange(x, 'b f g h w -> b (f g) h w')
        return x


# 计算外积的辅助函数
def _outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """torch实现的numpy的outer函数，用于计算1D向量的外积。"""
    a_flat = torch.reshape(a, [-1])
    b_flat = torch.reshape(b, [-1])
    a_mul = torch.unsqueeze(a_flat, dim=-1)
    b_mul = torch.unsqueeze(b_flat, dim=0)
    return a_mul * b_mul


# 构造二维滤波器的函数
def construct_2d_filt(lo, hi) -> torch.Tensor:
    """通过外积构造二维滤波器。

    参数：
        lo (torch.Tensor): 低通滤波器。
        hi (torch.Tensor): 高通滤波器。

    返回：
        torch.Tensor: 四个二维滤波器的堆叠（ll, lh, hl, hh）。
    """
    ll = _outer(lo, lo)
    lh = _outer(hi, lo)
    hl = _outer(lo, hi)
    hh = _outer(hi, hi)
    filt = torch.stack([ll, lh, hl, hh], 0)
    return filt


# 计算填充大小的函数
def _get_pad(data_len: int, filt_len: int) -> Tuple[int, int]:
    """计算所需的填充量。

    参数：
        data_len (int): 输入数据的长度。
        filt_len (int): 滤波器的长度。

    返回：
        tuple: 要加在输入数据两边的填充量。
    """
    padr = (2 * filt_len - 3) // 2
    padl = (2 * filt_len - 3) // 2

    # 对于奇数长度的数据，右边填充1
    if data_len % 2 != 0:
        padr += 1

    return padr, padl


# 对数据进行2D FWT填充的函数
def fwt_pad2(
        data: torch.Tensor, wavelet, mode: str = "replicate"
) -> torch.Tensor:
    """对数据进行2D FWT填充。

    参数：
        data (torch.Tensor): 输入数据（4维）。
        wavelet (Wavelet 或 str): pywt wavelet对象或wavelet名称。
        mode (str): 填充模式（支持'reflect', 'zero', 'constant', 'periodic'等模式，默认为'replicate'）。

    返回：
        torch.Tensor: 填充后的数据。
    """
    wavelet = _as_wavelet(wavelet)
    padb, padt = _get_pad(data.shape[-2], len(wavelet.dec_lo))
    padr, padl = _get_pad(data.shape[-1], len(wavelet.dec_lo))

    # 使用PyTorch的pad函数进行填充
    data_pad = F.pad(data, [padl, padr, padt, padb], mode=mode)
    return data_pad


# DWT（离散小波变换）模块
class DWT(nn.Module):
    def __init__(self, dec_lo, dec_hi, wavelet='haar', level=1, mode="replicate"):
        """
        初始化DWT类，定义小波滤波器和其他参数。

        参数:
        - dec_lo: 低频滤波器
        - dec_hi: 高频滤波器
        - wavelet: 小波类型（默认是Haar小波）
        - level: 小波分解的层数
        - mode: 填充模式（默认为"replicate"）
        """
        super(DWT, self).__init__()
        self.wavelet = _as_wavelet(wavelet)  # 将wavelet转换为小波对象
        self.dec_lo = dec_lo  # 低频滤波器
        self.dec_hi = dec_hi  # 高频滤波器
        self.level = level  # 小波分解的层数
        self.mode = mode  # 填充模式

    def forward(self, x):
        """
        执行小波变换。将输入图像进行多层小波分解。

        参数:
        - x: 输入的图像（形状为 [batch_size, channels, height, width]）

        返回:
        - wavelet_component: 小波变换后的结果，包含低频和高频分量
        """
        b, c, h, w = x.shape  # 获取输入图像的尺寸
        if self.level is None:
            self.level = pywt.dwtn_max_level([h, w], self.wavelet)  # 自动计算最大分解层数

        # 存储每一层的小波分量
        wavelet_component: List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []

        # 初始低频分量为输入图像
        l_component = x
        # 构造小波变换的卷积核
        dwt_kernel = construct_2d_filt(lo=self.dec_lo, hi=self.dec_hi)
        dwt_kernel = dwt_kernel.repeat(c, 1, 1)  # 对每个通道重复滤波器
        dwt_kernel = dwt_kernel.unsqueeze(dim=1)  # 添加额外的维度以便进行卷积操作

        # 进行多层小波分解
        for _ in range(self.level):
            l_component = fwt_pad2(l_component, self.wavelet, mode=self.mode)  # 填充并执行前向小波变换
            h_component = F.conv2d(l_component, dwt_kernel, stride=2, groups=c)  # 执行卷积以获取高频分量
            res = rearrange(h_component, 'b (c f) h w -> b c f h w', f=4)  # 重排列以分离低频和高频分量
            l_component, lh_component, hl_component, hh_component = res.split(1, 2)  # 分离低频和三个高频分量
            # 将高频分量存储到列表中
            wavelet_component.append((lh_component.squeeze(2), hl_component.squeeze(2), hh_component.squeeze(2)))

        # 添加最后的低频分量
        wavelet_component.append(l_component.squeeze(2))
        return wavelet_component[::-1]  # 反转顺序以从高频到低频返回结果


# IDWT（离散小波逆变换）模块
class IDWT(nn.Module):
    def __init__(self, rec_lo, rec_hi, wavelet='haar', level=1, mode="constant"):
        """
        初始化IDWT类，定义逆小波滤波器和其他参数。

        参数:
        - rec_lo: 低频重建滤波器
        - rec_hi: 高频重建滤波器
        - wavelet: 小波类型（默认是Haar小波）
        - level: 小波逆变换的层数
        - mode: 填充模式（默认为"constant"）
        """
        super(IDWT, self).__init__()
        self.rec_lo = rec_lo  # 低频重建滤波器
        self.rec_hi = rec_hi  # 高频重建滤波器
        self.wavelet = wavelet  # 小波类型
        self.level = level  # 小波逆变换的层数
        self.mode = mode  # 填充模式

    def forward(self, x, weight=None):
        """
        执行小波逆变换。根据输入的小波分量重建图像。

        参数:
        - x: 小波分量列表，包括低频和高频分量
        - weight: 可选的加权参数（默认为None，表示使用软正交）

        返回:
        - l_component: 重建后的图像
        """
        l_component = x[0]  # 取第一个元素作为初始低频分量
        _, c, _, _ = l_component.shape  # 获取通道数
        if weight is None:  # 如果没有指定权重，使用软正交
            idwt_kernel = construct_2d_filt(lo=self.rec_lo, hi=self.rec_hi)  # 构造逆小波滤波器
            idwt_kernel = idwt_kernel.repeat(c, 1, 1)  # 对每个通道重复滤波器
            idwt_kernel = idwt_kernel.unsqueeze(dim=1)  # 添加额外的维度以便进行卷积操作
        else:  # 如果指定了权重，使用硬正交
            idwt_kernel = torch.flip(weight, dims=[-1, -2])  # 对权重进行翻转

        # 进行小波逆变换
        self.filt_len = idwt_kernel.shape[-1]  # 获取滤波器长度
        for c_pos, component_lh_hl_hh in enumerate(x[1:]):  # 遍历每一层的小波分量
            # 将低频和高频分量拼接成一个tensor
            l_component = torch.cat([l_component.unsqueeze(2), component_lh_hl_hh[0].unsqueeze(2),
                                     component_lh_hl_hh[1].unsqueeze(2), component_lh_hl_hh[2].unsqueeze(2)], 2)
            l_component = rearrange(l_component, 'b c f h w -> b (c f) h w')  # 重排列

            # 执行卷积转置以重建图像
            l_component = F.conv_transpose2d(l_component, idwt_kernel, stride=2, groups=c)
            l_component = l_component.squeeze(1)  # 去除多余的维度

        return l_component  # 返回重建后的图像


class LWN(nn.Module):
    def __init__(self, dim, wavelet='haar', initialize=True, head=4, drop_rate=0., use_ca=False, use_sa=False):
        super(LWN, self).__init__()

        # 初始化参数
        self.dim = dim  # 输入特征的通道数
        self.wavelet = _as_wavelet(wavelet)  # 小波函数，转换为小波系数

        # 获取小波滤波器
        dec_lo, dec_hi, rec_lo, rec_hi = get_filter_tensors(wavelet, flip=True)

        # 根据initialize选择如何初始化小波滤波器
        if initialize:
            self.dec_lo = nn.Parameter(dec_lo, requires_grad=True)  # 分解低频滤波器
            self.dec_hi = nn.Parameter(dec_hi, requires_grad=True)  # 分解高频滤波器
            self.rec_lo = nn.Parameter(rec_lo.flip(-1), requires_grad=True)  # 重建低频滤波器，翻转
            self.rec_hi = nn.Parameter(rec_hi.flip(-1), requires_grad=True)  # 重建高频滤波器，翻转
        else:
            # 随机初始化滤波器
            self.dec_lo = nn.Parameter(torch.rand_like(dec_lo) * 2 - 1, requires_grad=True)
            self.dec_hi = nn.Parameter(torch.rand_like(dec_hi) * 2 - 1, requires_grad=True)
            self.rec_lo = nn.Parameter(torch.rand_like(rec_lo) * 2 - 1, requires_grad=True)
            self.rec_hi = nn.Parameter(torch.rand_like(rec_hi) * 2 - 1, requires_grad=True)

        # 定义DWT和IDWT模块
        self.wavedec = DWT(self.dec_lo, self.dec_hi, wavelet=wavelet, level=1)  # 离散小波变换（DWT）
        self.waverec = IDWT(self.rec_lo, self.rec_hi, wavelet=wavelet, level=1)  # 逆小波变换（IDWT）

        # 卷积层，conv1和conv2用来提取特征
        self.conv1 = nn.Conv2d(dim * 4, dim * 6, 1)  # 卷积1：从小波变换结果的4个分量提取特征
        self.conv2 = nn.Conv2d(dim * 6, dim * 6, 7, padding=3, groups=dim * 6)  # 卷积2：深度可分离卷积
        self.act = nn.GELU()  # 激活函数
        self.conv3 = nn.Conv2d(dim * 6, dim * 4, 1)  # 卷积3：将通道数压缩回原始数量

        # 是否使用空间注意力和通道注意力
        self.use_sa = use_sa
        self.use_ca = use_ca

        if self.use_sa:
            # 如果启用空间注意力，定义水平和垂直方向的空间注意力模块
            self.sa_h = nn.Sequential(
                nn.PixelShuffle(2),  # 上采样
                nn.Conv2d(dim // 4, 1, kernel_size=1, padding=0, stride=1, bias=True)  # 输出通道1
            )
            self.sa_v = nn.Sequential(
                nn.PixelShuffle(2),
                nn.Conv2d(dim // 4, 1, kernel_size=1, padding=0, stride=1, bias=True)
            )

        if self.use_ca:
            # 如果启用通道注意力，定义水平和垂直方向的通道注意力模块
            self.ca_h = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # 全局池化
                nn.Conv2d(dim, dim, 1, padding=0, stride=1, groups=1, bias=True),  # 1x1卷积
            )
            self.ca_v = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, dim, 1, padding=0, stride=1, groups=1, bias=True)
            )
            self.shuffle = ShuffleBlock(2)  # 用于通道重排的ShuffleBlock

    def forward(self, x):
        # 获取输入张量的形状
        _, _, H, W = x.shape

        # 执行小波分解，得到低频和高频分量
        ya, (yh, yv, yd) = self.wavedec(x)
        dec_x = torch.cat([ya, yh, yv, yd], dim=1)  # 将四个分量拼接起来

        # 通过卷积网络进行特征提取
        x = self.conv1(dec_x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)

        # 将输出分成四个部分
        ya, yh, yv, yd = torch.chunk(x, 4, dim=1)

        # 执行小波重建
        y = self.waverec([ya, (yh, yv, yd)], None)

        # 如果启用空间注意力，进行加权
        if self.use_sa:
            sa_yh = self.sa_h(yh)
            sa_yv = self.sa_v(yv)
            y = y * (sa_yv + sa_yh)  # 将加权后的特征与输出相乘

        # 如果启用通道注意力，进行加权
        if self.use_ca:
            # 通过上采样恢复较小的特征图
            yh = torch.nn.functional.interpolate(yh, scale_factor=2, mode='area')
            yv = torch.nn.functional.interpolate(yv, scale_factor=2, mode='area')

            # 计算通道注意力
            ca_yh = self.ca_h(yh)
            ca_yv = self.ca_v(yv)
            ca = self.shuffle(torch.cat([ca_yv, ca_yh], 1))  # 通道重排
            ca_1, ca_2 = ca.chunk(2, dim=1)  # 分割成两个部分
            ca = ca_1 * ca_2  # gated channel attention
            y = y * ca  # 将加权后的特征与输出相乘

        return y

    def get_wavelet_loss(self):
        # 返回小波重建损失
        return self.perfect_reconstruction_loss()[0] + self.alias_cancellation_loss()[0]

    def perfect_reconstruction_loss(self):
        """
        完美重建损失：确保小波分解和重建过程能够完美重建原始信号。
        理论上，滤波器应该满足P(z) + P(-z) = 2的条件。这里采用软约束。
        """
        # 计算P(z)的多项式乘积
        pad = self.dec_lo.shape[-1] - 1
        p_lo = F.conv1d(
            self.dec_lo.flip(-1).unsqueeze(0),
            self.rec_lo.flip(-1).unsqueeze(0),
            padding=pad)
        pad = self.dec_hi.shape[-1] - 1
        p_hi = F.conv1d(
            self.dec_hi.flip(-1).unsqueeze(0),
            self.rec_hi.flip(-1).unsqueeze(0),
            padding=pad)

        p_test = p_lo + p_hi

        two_at_power_zero = torch.zeros(p_test.shape, device=p_test.device,
                                        dtype=p_test.dtype)
        two_at_power_zero[..., p_test.shape[-1] // 2] = 2
        # 计算误差的平方
        errs = (p_test - two_at_power_zero) * (p_test - two_at_power_zero)
        return torch.sum(errs), p_test, two_at_power_zero

    def alias_cancellation_loss(self):
        """
        alias cancellation损失：确保小波滤波器满足F0(z)H0(-z) + F1(z)H1(-z) = 0的条件
        """
        m1 = torch.tensor([-1], device=self.dec_lo.device, dtype=self.dec_lo.dtype)
        length = self.dec_lo.shape[-1]
        mask = torch.tensor([torch.pow(m1, n) for n in range(length)][::-1],
                            device=self.dec_lo.device, dtype=self.dec_lo.dtype)
        # 计算多项式乘积
        pad = self.dec_lo.shape[-1] - 1
        p_lo = torch.nn.functional.conv1d(
            self.dec_lo.flip(-1).unsqueeze(0) * mask,
            self.rec_lo.flip(-1).unsqueeze(0),
            padding=pad)

        pad = self.dec_hi.shape[-1] - 1
        p_hi = torch.nn.functional.conv1d(
            self.dec_hi.flip(-1).unsqueeze(0) * mask,
            self.rec_hi.flip(-1).unsqueeze(0),
            padding=pad)

        p_test = p_lo + p_hi
        zeros = torch.zeros(p_test.shape, device=p_test.device,
                            dtype=p_test.dtype)
        errs = (p_test - zeros) * (p_test - zeros)
        return torch.sum(errs), p_test, zeros


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

class Bottleneck_LWN(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.cv3 = LWN(c2)
        self.add = shortcut and c1 == c2


    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))

class C2f_LWN(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck_LWN(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

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
    # 设置输入参数
    dim = 64  # 输入图像通道数
    batch_size = 4  # 批大小
    height = 32  # 输入图像的高度
    width = 25  # 输入图像的宽度

    # 创建一个随机输入图像张量
    x = torch.randn(batch_size, dim, height, width)

    # 初始化模型
    fft2_model = LWN(dim=dim)
    # fft3_model = FFT3(dim=dim)

    # 执行前向传播
    output_fft2 = fft2_model(x)
    # output_fft3 = fft3_model(x)

    # 打印输出的形状
    print("Input shape:", x.shape)
    print("Output shape (FFT2):", output_fft2.shape)
    # print("Output shape (FFT3):", output_fft3.shape)


if __name__ == "__main__":
    main()




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
#   - [-1, 3, C2f_WaveletConv, [128, True]]
#   - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
#   - [-1, 6, C2f_WaveletConv, [256, True]]
#   - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
#   - [-1, 6, C2f_WaveletConv, [512, True]]
#   - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
#   - [-1, 3, C2f_WaveletConv, [1024, True]]
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
