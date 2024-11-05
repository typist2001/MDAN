'''
This repository is used to implement all upsamplers(only x4) and tools for Efficient SR
@author
    LI Zehyuan from SIAT
    LIU yingqi from SIAT
'''

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import basicsr.archs.Upsamplers as Upsamplers
from basicsr.utils.registry import ARCH_REGISTRY


class DepthWiseConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_norm=False, bn_kwargs=None):
        super(DepthWiseConv, self).__init__()

        self.dw = torch.nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_ch,
            bias=bias,
            padding_mode=padding_mode,
        )

        self.pw = torch.nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

    def forward(self, input):
        out = self.dw(input)
        out = self.pw(out)
        return out


class BSConvU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.pw = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # depthwise
        self.dw = torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea


class BSConvS(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True,
                 padding_mode="zeros", p=0.25, min_mid_channels=4, with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        assert 0.0 <= p <= 1.0
        mid_channels = min(in_channels, max(min_mid_channels, math.ceil(p * in_channels)))
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise 1
        self.pw1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # pointwise 2
        self.add_module("pw2", torch.nn.Conv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        ))

        # depthwise
        self.dw = torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, x):
        fea = self.pw1(x)
        fea = self.pw2(fea)
        fea = self.dw(fea)
        return fea

    def _reg_loss(self):
        W = self[0].weight[:, :, 0, 0]
        WWt = torch.mm(W, torch.transpose(W, 0, 1))
        I = torch.eye(WWt.shape[0], device=WWt.device)
        return torch.norm(WWt - I, p="fro")


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True), nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0), nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# Multi-scale dilated attention block
class MDAB(nn.Module):
    def __init__(self, n_feats, conv=nn.Conv2d, p=0.25):
        super().__init__()
        kwargs = {'padding': 1}
        if conv.__name__ == 'BSConvS':
            kwargs = {'p': p}

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        self.b1 = conv(n_feats, n_feats, kernel_size=3, **kwargs)

        self.b1_1 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 5, stride=1, padding=(5 // 2) * 2, groups=n_feats // 3, dilation=2),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))
        self.b1_2 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))
        self.b1_3 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0),
            nn.Conv2d(n_feats // 3, n_feats // 3, 7, stride=1, padding=(7 // 2) * 3, groups=n_feats // 3, dilation=3)
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))

        self.conv1_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))

    def forward(self, x):
        shortcut = x.clone()

        x = self.norm(x)

        f1 = self.b1(x)

        f2 = self.b2(x)

        f1_1, f1_2, f1_3 = torch.chunk(f1, 3, dim=1)

        f1 = torch.cat([self.b1_1(f1_1), self.b1_2(f1_2), self.b1_3(f1_3)], dim=1)
        visual(self.b1_1(f1_1), 'b1')
        visual(self.b1_2(f1_2), 'b2')
        visual(self.b1_3(f1_3), 'b3')
        visual(f1, 'att')
        x = self.conv1_last(f1 * f2) * self.scale + shortcut
        visual(self.conv1_last(f1 * f2),'f12')
        visual(x, 'mdab')
        return x

# 模型中间结果可视化
num_of_label = {}
out_dir = r'C:\Users\ghhgy\Desktop\out\\'

import matplotlib.pyplot as plt
def visual(x, label):
    if not label in num_of_label:
        num_of_label[label] = 0
    temp = x.cpu()[0]
    # 转换为NumPy数组
    temp = temp.float().data.numpy()
    # 将所有维度叠加在一起，并将结果转换为灰度图像
    combined_image = temp.sum(axis=0)  # 将所有维度相加
    combined_image = (combined_image - combined_image.min()) / (
            combined_image.max() - combined_image.min())  # 将像素值归一化到0到1之间
    # 绘制图像
    plt.figure(figsize=(12, 8))  # 设置图像大小
    plt.imshow(combined_image)
    plt.colorbar()  # 增加colorbar图例
    # plt.show()
    plt.savefig(out_dir + f'{label}_{num_of_label[label]}.png')
    num_of_label[label] += 1


# Multi-scale dilated attention module
class MDAM(nn.Module):
    def __init__(self, n_feats, conv=nn.Conv2d, p=0.25):
        super().__init__()

        self.MDAB = MDAB(n_feats, conv, p)
        self.BFN = BFN(n_feats)

    def forward(self, x):
        x = self.MDAB(x)
        x = self.BFN(x)
        visual(x,'bfn')
        return x


# Blueprint feed-forward network
class BFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=1, bias=False):
        super().__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=7, stride=1, padding=7 // 2,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        # norm
        self.norm = LayerNorm(dim, data_format='channels_first')

    def forward(self, x):
        shortcut = x.clone()
        x = self.project_in(self.norm(x))
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x) + shortcut
        return x


# Local residual module
class LRM(nn.Module):
    def __init__(self, n_feats, conv=nn.Conv2d, p=0.25):
        super().__init__()
        kwargs = {'padding': 1}
        if conv.__name__ == 'BSConvS':
            kwargs = {'p': p}

        self.lbc1 = conv(n_feats, n_feats, kernel_size=3, **kwargs)
        self.lbc2 = conv(n_feats, n_feats, kernel_size=3, **kwargs)
        self.lbc3 = conv(n_feats, n_feats, kernel_size=3, **kwargs)
        self.act = nn.GELU()
        self.conv1 = nn.Conv2d(n_feats, n_feats, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.act(self.lbc1(x))
        x = self.act(self.lbc2(x))
        x = self.act(self.lbc3(x))
        x = x + shortcut
        x = self.conv1(x)
        return x


# Feature refinement module
class FRM(nn.Module):
    def __init__(self, n_feats, conv=nn.Conv2d, p=0.25):
        super().__init__()
        kwargs = {'padding': 1}
        if conv.__name__ == 'BSConvS':
            kwargs = {'p': p}

        self.LRM = LRM(n_feats, conv, p)
        self.MDAM = MDAM(n_feats, conv, p)

    def forward(self, x):
        x = self.LRM(x)
        x = self.MDAM(x)
        return x


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)





@ARCH_REGISTRY.register()
class MDAN_visual(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=51, num_block=8, num_out_ch=3, upscale=4,
                 conv='BSConvU', upsampler='pixelshuffledirect', p=0.25):
        super(MDAN_visual, self).__init__()
        kwargs = {'padding': 1}
        if conv == 'BSConvS':
            kwargs = {'p': p}
        print(conv)
        if conv == 'DepthWiseConv':
            self.conv = DepthWiseConv
        elif conv == 'BSConvU':
            self.conv = BSConvU
        elif conv == 'BSConvS':
            self.conv = BSConvS
        else:
            self.conv = nn.Conv2d
        self.fea_conv = self.conv(num_in_ch * 4, num_feat, kernel_size=3, **kwargs)

        self.B1 = FRM(n_feats=num_feat, conv=self.conv, p=p)
        self.B2 = FRM(n_feats=num_feat, conv=self.conv, p=p)
        self.B3 = FRM(n_feats=num_feat, conv=self.conv, p=p)
        self.B4 = FRM(n_feats=num_feat, conv=self.conv, p=p)
        self.B5 = FRM(n_feats=num_feat, conv=self.conv, p=p)
        self.B6 = FRM(n_feats=num_feat, conv=self.conv, p=p)
        self.B7 = FRM(n_feats=num_feat, conv=self.conv, p=p)
        self.B8 = FRM(n_feats=num_feat, conv=self.conv, p=p)

        self.c1 = nn.Conv2d(num_feat * num_block, num_feat, 1)
        self.GELU = nn.GELU()

        self.c2 = self.conv(num_feat, num_feat, kernel_size=3, **kwargs)

        if upsampler == 'pixelshuffledirect':
            self.upsampler = Upsamplers.PixelShuffleDirect(scale=upscale, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'pixelshuffleblock':
            self.upsampler = Upsamplers.PixelShuffleBlcok(in_feat=num_feat, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'nearestconv':
            self.upsampler = Upsamplers.NearestConv(in_ch=num_feat, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'pa':
            self.upsampler = Upsamplers.PA_UP(nf=num_feat, unf=24, out_nc=num_out_ch)
        else:
            raise NotImplementedError(("Check the Upsampeler. None or not support yet"))

    def forward(self, input):
        input = torch.cat([input, input, input, input], dim=1)
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        out_B7 = self.B7(out_B6)
        out_B8 = self.B8(out_B7)

        trunk = torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8], dim=1)
        out_B = self.c1(trunk)
        out_B = self.GELU(out_B)

        out_lr = self.c2(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output
