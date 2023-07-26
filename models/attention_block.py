from models.HRNet import conv3x3, BasicBlock
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
BN_MOMENTUM = 0.01


class CSAMBlock(nn.Module):
    def __init__(self, out_feat=48, stage_channels=[48, 96, 192, 382], attention='scam'):
        super(CSAMBlock, self).__init__()
        self.check_layer = nn.Sequential(
            conv3x3(1, out_feat, stride=1),
            nn.BatchNorm2d(out_feat, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            conv3x3(out_feat, out_feat, stride=1),
            nn.BatchNorm2d(out_feat, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        # print(stage_channels)
        self.stages = nn.ModuleList([MSCSAM(ic, out_feat, attention=attention) for ic in stage_channels])

        self.dfl = nn.Sequential(
            conv3x3(out_feat * len(stage_channels), out_feat, stride=1),
            nn.BatchNorm2d(out_feat, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))

    def forward(self, y_list, check_map):
        c = self.check_layer(check_map)
        x0_h, x0_w = y_list[0].size(2), y_list[0].size(3)
        list_feature = []
        for i, f in enumerate(y_list):
            x = self.stages[i](c, f)
            list_feature.append(F.upsample(x, size=(x0_h, x0_w), mode='bilinear', align_corners=True))
        x = self.dfl(torch.cat(list_feature, dim=1))
        return x


class MSCSAM(nn.Module):
    def __init__(self, in_channels, out_channels, attention='scam'):
        super().__init__()
        self.print = (in_channels, out_channels)
        self.feature_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.res_fuse_layer = nn.Sequential(
            conv3x3(out_channels, out_channels, stride=1),
            nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        if attention == 'scam':
            self.attention = CSAM(inchannels=out_channels)

    def forward(self, c, f):
        _, _, m, n = f.shape
        f = self.feature_layer(f)
        x = self.res_fuse_layer(f - F.interpolate(c, size=(m, n), mode='bilinear'))
        x = self.attention(x)
        return x


class CSAM(nn.Module):
    def __init__(self, inchannels, kernel_size=7):
        super().__init__()
        self.CA = ChannelAttention(in_planes=inchannels)
        self.SA = SpatialAttention(kernel_size=kernel_size)

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inchannels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannels, inchannels, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        ca = self.CA(x) * x
        sa = self.SA(x) * x
        out = x + ca + sa
        out = self.conv1x1(out)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid_planes = max(64, in_planes//2)
        self.fc1   = nn.Conv2d(in_planes, mid_planes, 1, bias=False)
        self.relu1 = nn.functional.leaky_relu
        self.fc2   = nn.Conv2d(mid_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)