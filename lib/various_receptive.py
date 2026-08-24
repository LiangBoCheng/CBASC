import torch
from torch import mean, nn
from collections import OrderedDict
from torch.nn import functional as F
import numpy as np
from numpy import random
import os


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 2, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 2, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], dim=1)
        x2 = self.conv1(x1)
        return self.sigmoid(x2)

class DirectionWeight(nn.Module):
    """
    Global Direction Selection
    Input:
        x1,x2,x3,x4 : [B,C,H,W]

    Output:
        weight : [B,4]
    """

    def __init__(self, channel, reduction=16):
        super(DirectionWeight, self).__init__()

        hidden = max(channel // reduction, 4)

        self.fc = nn.Sequential(
            nn.Linear(channel * 4, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 4)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3, x4):

        B, C, _, _ = x1.shape

        z1 = F.adaptive_avg_pool2d(x1, 1).view(B, C)
        z2 = F.adaptive_avg_pool2d(x2, 1).view(B, C)
        z3 = F.adaptive_avg_pool2d(x3, 1).view(B, C)
        z4 = F.adaptive_avg_pool2d(x4, 1).view(B, C)

        z = torch.cat([z1, z2, z3, z4], dim=1)

        weight = self.fc(z)

        weight = self.softmax(weight)

        return weight


class DAF(nn.Module):
    def __init__(self, channel):
        super(DAF, self).__init__()

        # Horizontal
        self.h_conv = BasicConv2d(channel, channel, kernel_size=(1, 5), padding=(0, 2))

        # Vertical
        self.w_conv = BasicConv2d(channel, channel, kernel_size=(5, 1), padding=(2, 0))

        # Leading diagonal
        self.dia19_conv = BasicConv2d(channel, channel, kernel_size=(5, 1), padding=(2, 0))

        # Reverse diagonal
        self.dia37_conv = BasicConv2d(channel, channel, kernel_size=(1, 5), padding=(0, 2))

        # Direction Selection
        self.direction_weight = DirectionWeight(channel)

        # Fusion Conv
        self.conv_fuse = BasicConv2d(channel, channel, kernel_size=3, padding=1)

    def forward(self, x):

        residual = x

        # Horizontal
        x1 = self.h_conv(x)
        # Vertical
        x2 = self.w_conv(x)
        # Leading diagonal
        x3 = self.inv_h_transform(self.dia19_conv(self.h_transform(x)))
        # Reverse diagonal
        x4 = self.inv_v_transform(self.dia37_conv(self.v_transform(x)))
        # ====================================================
        # Global Direction Selection
        # ====================================================
        weight = self.direction_weight(x1,x2,x3,x4)  # [B,4]

        w1 = weight[:, 0].view(-1, 1, 1, 1)
        w2 = weight[:, 1].view(-1, 1, 1, 1)
        w3 = weight[:, 2].view(-1, 1, 1, 1)
        w4 = weight[:, 3].view(-1, 1, 1, 1)

        fusion = w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4

        out = self.conv_fuse(fusion)

        out = out + residual

        return out

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-2]]
        x = x.reshape(shape[0], shape[1], shape[2], shape[2]+shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[2], shape[3]+1)
        x = x[..., 0: shape[3]-shape[2]+1]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-2]]
        x = x.reshape(shape[0], shape[1], shape[2], shape[2]+shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[2], shape[3]+1)
        x = x[..., 0: shape[3]-shape[2]+1]
        return x.permute(0, 1, 3, 2)

class Direction_aware_Adaptive_Fusion(nn.Module):
    def __init__(self, dim):
        super(Direction_aware_Adaptive_Fusion, self).__init__()
        self.dirConv = DAF(dim)

    def forward(self, x):
        bs, n, dim = x.shape
        h, w = int(np.sqrt(n)), int(np.sqrt(n))

        input = x.view(bs, h, w, dim).permute(0, 3, 1, 2)  # bs,dim,h,w
        out = self.dirConv(input)  # bs,dim,h,w
        out = out.reshape(bs, dim, -1).permute(0, 2, 1)  # bs,h*w,dim
        return out