import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bn=True, relu=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.data_format = data_format
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mean = x.mean((1, 2, 3), keepdim=True)
        std = x.std((1, 2, 3), keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd for 'same' padding."
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        pool = torch.cat([max_pool, avg_pool], dim=1)
        out = self.conv(pool)
        return self.sigmoid(out)

class MM_2Block(nn.Module):
    def __init__(self, ch_1, ch_2, drop_rate=0.):
        super(MM_2Block, self).__init__()
        self.drop_rate = drop_rate
        out_channels = (ch_1 + ch_2)

        self.channel_attention = Conv(ch_1 + ch_2, ch_1 + ch_2, 1, bn=True, relu=True)
        self.spatial_attention = SpatialAttention(kernel_size=7)

        self.combine_and_expand = Conv(ch_1 + ch_2, out_channels, 1, bn=True, relu=True)

        self.dropout = nn.Dropout2d(drop_rate)

    def forward(self, g, x):
        gx_combined = torch.cat([g, x], dim=1)
        gx_att = self.channel_attention(gx_combined)

        spatial_att_map = self.spatial_attention(gx_combined)
        gx_spatial_att = gx_combined * spatial_att_map

        out = self.combine_and_expand(gx_spatial_att)

        if self.drop_rate > 0:
            out = self.dropout(out)

        return out
