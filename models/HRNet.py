# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)


stage1_18_cfg = {'NUM_CHANNELS': [64], 'BLOCK':'BOTTLENECK', 'NUM_BLOCKS':[4], 'NUM_MODULES':1, 'NUM_BRANCHES':1, 'FUSE_METHOD':'SUM'}
stage2_18_cfg = {'NUM_CHANNELS': [18,36], 'BLOCK':'BASIC', 'NUM_BLOCKS':[4,4], 'NUM_MODULES':1, 'NUM_BRANCHES':2, 'FUSE_METHOD':'SUM'}
stage3_18_cfg = {'NUM_CHANNELS': [18,36,72], 'BLOCK':'BASIC', 'NUM_BLOCKS':[4,4,4], 'NUM_MODULES':4, 'NUM_BRANCHES':3, 'FUSE_METHOD':'SUM'}
stage4_18_cfg = {'NUM_CHANNELS': [18,36,72,144], 'BLOCK':'BASIC', 'NUM_BLOCKS':[4,4,4,4], 'NUM_MODULES':3, 'NUM_BRANCHES':4, 'FUSE_METHOD':'SUM'}
hrnet_w18_cfg = {'stage1':stage1_18_cfg,'stage2':stage2_18_cfg,'stage3':stage3_18_cfg,'stage4':stage4_18_cfg}

stage1_30_cfg = {'NUM_CHANNELS': [64], 'BLOCK':'BOTTLENECK', 'NUM_BLOCKS':[4], 'NUM_MODULES':1, 'NUM_BRANCHES':1, 'FUSE_METHOD':'SUM'}
stage2_30_cfg = {'NUM_CHANNELS': [30,60], 'BLOCK':'BASIC', 'NUM_BLOCKS':[4,4], 'NUM_MODULES':1, 'NUM_BRANCHES':2, 'FUSE_METHOD':'SUM'}
stage3_30_cfg = {'NUM_CHANNELS': [30,60,120], 'BLOCK':'BASIC', 'NUM_BLOCKS':[4,4,4], 'NUM_MODULES':4, 'NUM_BRANCHES':3, 'FUSE_METHOD':'SUM'}
stage4_30_cfg = {'NUM_CHANNELS': [30,60,120,240], 'BLOCK':'BASIC', 'NUM_BLOCKS':[4,4,4,4], 'NUM_MODULES':3, 'NUM_BRANCHES':4, 'FUSE_METHOD':'SUM'}
hrnet_w30_cfg = {'stage1':stage1_30_cfg,'stage2':stage2_30_cfg,'stage3':stage3_30_cfg,'stage4':stage4_30_cfg}

stage1_40_cfg = {'NUM_CHANNELS': [64], 'BLOCK':'BOTTLENECK', 'NUM_BLOCKS':[4], 'NUM_MODULES':1, 'NUM_BRANCHES':1, 'FUSE_METHOD':'SUM'}
stage2_40_cfg = {'NUM_CHANNELS': [40,80], 'BLOCK':'BASIC', 'NUM_BLOCKS':[4,4], 'NUM_MODULES':1, 'NUM_BRANCHES':2, 'FUSE_METHOD':'SUM'}
stage3_40_cfg = {'NUM_CHANNELS': [40,80,160], 'BLOCK':'BASIC', 'NUM_BLOCKS':[4,4,4], 'NUM_MODULES':4, 'NUM_BRANCHES':3, 'FUSE_METHOD':'SUM'}
stage4_40_cfg = {'NUM_CHANNELS': [40,80,160,320], 'BLOCK':'BASIC', 'NUM_BLOCKS':[4,4,4,4], 'NUM_MODULES':3, 'NUM_BRANCHES':4, 'FUSE_METHOD':'SUM'}
hrnet_w40_cfg = {'stage1':stage1_40_cfg,'stage2':stage2_40_cfg,'stage3':stage3_40_cfg,'stage4':stage4_40_cfg}

stage1_48_cfg = {'NUM_CHANNELS': [64], 'BLOCK':'BOTTLENECK', 'NUM_BLOCKS':[4], 'NUM_MODULES':1, 'NUM_BRANCHES':1, 'FUSE_METHOD':'SUM'}
stage2_48_cfg = {'NUM_CHANNELS': [48,96], 'BLOCK':'BASIC', 'NUM_BLOCKS':[4,4], 'NUM_MODULES':1, 'NUM_BRANCHES':2, 'FUSE_METHOD':'SUM'}
stage3_48_cfg = {'NUM_CHANNELS': [48,96,192], 'BLOCK':'BASIC', 'NUM_BLOCKS':[4,4,4], 'NUM_MODULES':4, 'NUM_BRANCHES':3, 'FUSE_METHOD':'SUM'}
stage4_48_cfg = {'NUM_CHANNELS': [48,96,192,384], 'BLOCK':'BASIC', 'NUM_BLOCKS':[4,4,4,4], 'NUM_MODULES':3, 'NUM_BRANCHES':4, 'FUSE_METHOD':'SUM'}
hrnet_w48_cfg = {'stage1':stage1_48_cfg,'stage2':stage2_48_cfg,'stage3':stage3_48_cfg,'stage4':stage4_48_cfg}

stage1_64_cfg = {'NUM_CHANNELS': [64], 'BLOCK':'BOTTLENECK', 'NUM_BLOCKS':[4], 'NUM_MODULES':1, 'NUM_BRANCHES':1, 'FUSE_METHOD':'SUM'}
stage2_64_cfg = {'NUM_CHANNELS': [64,128], 'BLOCK':'BASIC', 'NUM_BLOCKS':[4,4], 'NUM_MODULES':1, 'NUM_BRANCHES':2, 'FUSE_METHOD':'SUM'}
stage3_64_cfg = {'NUM_CHANNELS': [64,128,256], 'BLOCK':'BASIC', 'NUM_BLOCKS':[4,4,4], 'NUM_MODULES':4, 'NUM_BRANCHES':3, 'FUSE_METHOD':'SUM'}
stage4_64_cfg = {'NUM_CHANNELS': [64,128,256,512], 'BLOCK':'BASIC', 'NUM_BLOCKS':[4,4,4,4], 'NUM_MODULES':3, 'NUM_BRANCHES':4, 'FUSE_METHOD':'SUM'}
hrnet_w64_cfg = {'stage1':stage1_64_cfg,'stage2':stage2_64_cfg,'stage3':stage3_64_cfg,'stage4':stage4_64_cfg}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3, 
                                            momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.upsample(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=True)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck,
}


class ResFeatureBlock(nn.Module):
    def __init__(self, in_feat, out_feat=48):
        super(ResFeatureBlock, self).__init__()
        self.check_layer = nn.Sequential(
            conv3x3(1, out_feat, stride=1),
            nn.BatchNorm2d(out_feat, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_feat, out_feat, kernel_size=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )

        self.feature_layer = nn.Conv2d(in_feat, out_feat, kernel_size=1)

        self.res_fuse_layer = nn.Sequential(
            conv3x3(out_feat, out_feat, stride=1),
            nn.BatchNorm2d(out_feat, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            BasicBlock(out_feat, out_feat),
            BasicBlock(out_feat, out_feat),
            BasicBlock(out_feat, out_feat)
        )

    def forward(self, feature, check_map):
        f = self.feature_layer(feature)
        c = self.check_layer(check_map)
        x = self.res_fuse_layer(f - c)
        return x


class HRNet(nn.Module):

    def __init__(self, in_channels=3, num_classes=1, hr_cfg='w48', with_upsample=True):
        super(HRNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        if hr_cfg=='w64':
            hrnet_cfg = hrnet_w64_cfg
        if hr_cfg=='w48':
            hrnet_cfg = hrnet_w48_cfg
        if hr_cfg=='w40':
            hrnet_cfg = hrnet_w40_cfg
        if hr_cfg=='w30':
            hrnet_cfg = hrnet_w30_cfg
        if hr_cfg=='w18':
            hrnet_cfg = hrnet_w18_cfg     
        self.stage1_cfg = hrnet_cfg['stage1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels

        self.stage2_cfg = hrnet_cfg['stage2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = hrnet_cfg['stage3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = hrnet_cfg['stage4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        self.stage_channels = pre_stage_channels
        last_inp_channels = int(np.sum(pre_stage_channels))

        self.input_channels = last_inp_channels

        self.fusion = nn.Sequential(
            nn.Conv2d(last_inp_channels,last_inp_channels,5,1,2),
            nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

        self.seg_head = nn.Sequential(
            nn.Conv2d(last_inp_channels, num_classes, 1, 1, 0),
            nn.Sigmoid()
        )
        self.with_upsample = with_upsample

        self.init_weights()

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)


    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)


    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                      block,
                                      num_blocks,
                                      num_inchannels,
                                      num_channels,
                                      fuse_method,
                                      reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def base_forward(self, x):
        _, c, m, n = x.shape
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        # Upsampling
        x0_h, x0_w = y_list[0].size(2), y_list[0].size(3)

        x0 = F.upsample(y_list[0], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x1 = F.upsample(y_list[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.upsample(y_list[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.upsample(y_list[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x = torch.cat([x0, x1, x2, x3], 1)
        x = self.fusion(x)
        binary = self.seg_head(x)
        if self.with_upsample:
            binary = F.upsample(input=binary, size=(m, n), mode='bilinear', align_corners=True)

        return binary, x

    def forward(self, x):
        return self.base_forward(x)

    def init_weights(self, pretrained=''):
        # logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pre_weight = torch.load(pretrained)
            # logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {}
            old_conv1_weight = pre_weight['conv1.weight']
            channel = model_dict['conv1.weight'].size()[1]
            num = 0
            for i in range(0,channel):
                model_dict['conv1.weight'][:,i,:,:] = old_conv1_weight[:,i%3,:,:]
                num += 1

            for k, v in pre_weight.items():
                if k in model_dict.keys() and k!= 'conv1.weight':
                    pretrained_dict[k] = v
                    num += 1
            model_dict.update(pretrained_dict)
            print(f'load-weight:{num}/{len(pre_weight.keys())}')
            self.load_state_dict(model_dict)


class Baseline(HRNet):
    def __init__(self, in_channels=3, num_classes=1, hr_cfg='w48'):
        super(Baseline, self).__init__(in_channels, num_classes, hr_cfg)

        input_channels = self.input_channels

        self.seg_head = nn.Sequential(
            nn.Conv2d(input_channels, num_classes, 1, 1, 0),
            nn.Sigmoid()
        )

    def base_forward(self, x):
        _, c, m, n = x.shape
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        # Upsampling
        x0_h, x0_w = y_list[0].size(2), y_list[0].size(3)
        x0 = F.upsample(y_list[0], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x1 = F.upsample(y_list[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.upsample(y_list[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.upsample(y_list[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x = torch.cat([x0, x1, x2, x3], 1)

        x = self.fusion(x)
        binary = self.seg_head(x)
        binary = F.upsample(input=binary, size=(m, n), mode='bilinear', align_corners=True)
        return binary

    def forward(self, x):
        return self.base_forward(x)

    def init_weights(self, pretrained='', model_pretrained=''):
        # logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pre_weight = torch.load(pretrained)
            # logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {}
            old_conv1_weight = pre_weight['conv1.weight']
            channel = model_dict['conv1.weight'].size()[1]
            num = 0
            for i in range(0, channel):
                model_dict['conv1.weight'][:, i, :, :] = old_conv1_weight[:, i % 3, :, :]
                num += 1

            for k, v in pre_weight.items():
                if k in model_dict.keys() and k != 'conv1.weight':
                    pretrained_dict[k] = v
                    num += 1
            model_dict.update(pretrained_dict)
            print(f'load-weight:{num}/{len(pre_weight.keys())}')
            self.load_state_dict(model_dict)

        if os.path.isfile(model_pretrained):
            pre_weight = torch.load(model_pretrained, map_location='cpu')

            model_dict = self.state_dict()
            num = 0
            for k, v in pre_weight.items():
                k = k[6:]
                if k in model_dict.keys():
                    model_dict[k] = v
                    num += 1
            print(f'model-load-weight:{num}/{len(pre_weight.keys())}')
            self.load_state_dict(model_dict)


class HRNetV2(nn.Module):

    def __init__(self, in_channels=3, hr_cfg='w48'):
        super(HRNetV2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        if hr_cfg == 'w64':
            hrnet_cfg = hrnet_w64_cfg
        if hr_cfg == 'w48':
            hrnet_cfg = hrnet_w48_cfg
        if hr_cfg == 'w40':
            hrnet_cfg = hrnet_w40_cfg
        if hr_cfg == 'w30':
            hrnet_cfg = hrnet_w30_cfg
        if hr_cfg == 'w18':
            hrnet_cfg = hrnet_w18_cfg
        self.stage1_cfg = hrnet_cfg['stage1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = hrnet_cfg['stage2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = hrnet_cfg['stage3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = hrnet_cfg['stage4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        self.stage_channels = pre_stage_channels

        self.init_weights()

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def base_forward(self, x):
        _, c, m, n = x.shape
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        return y_list

    def forward(self, x):
        return self.base_forward(x)

    def init_weights(self, pretrained=''):
        # logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pre_weight = torch.load(pretrained)
            # logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {}
            old_conv1_weight = pre_weight['conv1.weight']
            channel = model_dict['conv1.weight'].size()[1]
            num = 0
            for i in range(0, channel):
                model_dict['conv1.weight'][:, i, :, :] = old_conv1_weight[:, i % 3, :, :]
                num += 1

            for k, v in pre_weight.items():
                if k in model_dict.keys() and k != 'conv1.weight':
                    pretrained_dict[k] = v
                    num += 1
            model_dict.update(pretrained_dict)
            print(f'load-weight:{num}/{len(pre_weight.keys())}')
            self.load_state_dict(model_dict)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    model = HRNet(in_channels=3, hr_cfg='w48').cuda()
    model.init_weights(pretrained='/data02/zzl/codes2022/DataQualityChecking/torch_cache/hrnetv2_w48_imagenet_pretrained.pth')
    inp = torch.randn(2, 3, 512, 512).cuda()
    check_map = torch.randn(2, 1, 512, 512).cuda()

    with torch.no_grad():
        out, out_aux = model(inp)
        print(out.size(), out_aux.size())
