import math
from os.path import join as pjoin
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .concat import MM_3block, MultiplicativeFusion, Conv,AdditiveFusion


def np2th(weights, conv=False):

    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):


        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)


        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)


        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel").replace("\\", "/")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel").replace("\\", "/")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel").replace("\\", "/")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale").replace("\\", "/")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias").replace("\\", "/")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale").replace("\\", "/")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias").replace("\\", "/")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale").replace("\\", "/")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias").replace("\\", "/")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel").replace("\\", "/")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale").replace("\\", "/")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias").replace("\\", "/")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))


class ResNetV2(nn.Module):

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width
        self.fu0 = MultiplicativeFusion()
        self.Updim = Conv(64, 128, 1, bn=True, relu=True)
        self.fu1 = MM_3block(ch_1=256, ch_2=256, r_2=4, ch_int=256, ch_out=256, drop_rate=0.1)
        self.fu2 = MM_3block(ch_1=512, ch_2=512, r_2=8, ch_int=512, ch_out=512, drop_rate=0.1)
        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x, y):
        features_x = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features_x.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features_x.append(feat)
        features_y = []
        b, c, in_size, _ = y.size()
        y = self.root(y)
        features_y.append(y)
        y = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(y)
        for i in range(len(self.body) - 1):
            y = self.body[i](y)
            right_size = int(in_size / 4 / (i+1))
            if y.size()[2] != right_size:
                pad = right_size - y.size()[2]
                assert pad < 3 and pad > 0, "y {} should {}".format(y.size(), right_size)
                feat = torch.zeros((b, y.size()[1], right_size, right_size), device=y.device)
                feat[:, :, 0:y.size()[2], 0:y.size()[3]] = y[:]
            else:
                feat = y
            features_y.append(feat)
        additive_fusion = AdditiveFusion()
        features = []
        features.append(self.fu0(features_x[0], features_y[0]))
        f = self.Updim(features[0])
        features.append(self.fu1(features_x[1], features_y[1], f))
        features.append(self.fu2(features_x[2], features_y[2], features[-1]))
        x = self.body[-1](x)
        return x, features[::-1]
