# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule, kaiming_init
from mmcv.runner import load_checkpoint
from torchinfo import summary
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


# Datashape num_joints x num_frames x 2D coord. x num_pers
NUM_FRAMES_MIN = 32
NUM_PERS_MAX = 2
COORD_2D = 2


class C3dModifiedJointFramePersDim(nn.Module):
    """C3D backbone, with flatten and mlp.

    Args:
        pretrained (str | None): Name of pretrained model.
    """

    def __init__(self,
                 in_channels=17,
                 base_channels=64,
                 num_stages=4,
                 temporal_downsample=True,
                 pretrained=None):
        super().__init__()
        conv_cfg = dict(type='Conv3d')
        norm_cfg = dict(type='BN3d')
        act_cfg = dict(type='ReLU')
        self.pretrained = pretrained
        self.in_channels = in_channels
        self.base_channels = base_channels
        assert num_stages in [3, 4]
        self.num_stages = num_stages
        self.temporal_downsample = temporal_downsample
        pool_kernel, pool_stride = (3, 2, 2), (2, 1, 1)
        if not self.temporal_downsample:
            pool_kernel, pool_stride = (3, 2, 2), (2, 1, 1)

        c3d_conv_param = dict(kernel_size=(3, 2, 2), padding=(1, 1, 1), conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.conv1a = ConvModule(self.in_channels, self.base_channels, **c3d_conv_param)
        #print(self.conv1a.weight.dtype)
        self.pool1 = nn.AvgPool3d(kernel_size=(3, 2, 2), stride=(2, 1, 1))

        self.conv2a = ConvModule(self.base_channels, self.base_channels * 2, **c3d_conv_param)
        self.pool2 = nn.AvgPool3d(kernel_size=pool_kernel, stride=pool_stride)

        self.conv3a = ConvModule(self.base_channels * 2, self.base_channels * 4, **c3d_conv_param)
        self.conv3b = ConvModule(self.base_channels * 4, self.base_channels * 4, **c3d_conv_param)
        self.pool3 = nn.AvgPool3d(kernel_size=pool_kernel, stride=pool_stride)

        self.conv4a = ConvModule(self.base_channels * 4, self.base_channels * 8, **c3d_conv_param)
        self.conv4b = ConvModule(self.base_channels * 8, self.base_channels * 8, **c3d_conv_param)

        if self.num_stages == 4:
            self.pool4 = nn.AvgPool3d(kernel_size=pool_kernel, stride=pool_stride)
            self.conv5a = ConvModule(self.base_channels * 8, self.base_channels * 8, **c3d_conv_param)
            self.conv5b = ConvModule(self.base_channels * 8, self.base_channels * 8, **c3d_conv_param)
        
        self.fc1 = nn.Linear(18432, 512)
        self.fc2 = nn.Linear(512, 60)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data. The size of x is (num_batches, 3, 16, 112, 112).

        Returns:
            torch.Tensor: The feature of the input samples extracted by the backbone.
        """
        x = self.conv1a(x)

        x = self.pool1(x)
        x = self.conv2a(x)

        x = self.pool2(x)
        x = self.conv3a(x)
        x = self.conv3b(x)

        x = self.pool3(x)
        x = self.conv4a(x)
        x = self.conv4b(x)

        if self.num_stages == 3:
            return x

        x = self.pool4(x)
        x = self.conv5a(x)
        x = self.conv5b(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.fc2(x)

        x = F.softmax(x, dim=1)

        return x


class C3dModifiedPersJointFrameDim(nn.Module):
    """C3D backbone, with flatten and mlp.

    Args:
        pretrained (str | None): Name of pretrained model.
    """

    def __init__(self,
                 in_channels=2,
                 base_channels=64,
                 num_stages=4,
                 temporal_downsample=True,
                 pretrained=None):
        super().__init__()
        conv_cfg = dict(type='Conv3d')
        norm_cfg = dict(type='BN3d')
        act_cfg = dict(type='ReLU')
        self.pretrained = pretrained
        self.in_channels = in_channels
        self.base_channels = base_channels
        assert num_stages in [3, 4]
        self.num_stages = num_stages
        self.temporal_downsample = temporal_downsample
        pool_kernel, pool_stride = (3, 2, 2), (2, 2, 1)
        if not self.temporal_downsample:
            print('hey brUAW')
            pool_kernel, pool_stride = (3, 2, 2), (2, 2, 1)

        c3d_conv_param = dict(kernel_size=(3, 2, 2), padding=(1, 1, 1), conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.conv1a = ConvModule(self.in_channels, self.base_channels, **c3d_conv_param)
        self.pool1 = nn.AvgPool3d(kernel_size=(3, 2, 2), stride=(2, 2, 1))

        self.conv2a = ConvModule(self.base_channels, self.base_channels * 2, **c3d_conv_param)
        self.pool2 = nn.AvgPool3d(kernel_size=pool_kernel, stride=pool_stride)

        self.conv3a = ConvModule(self.base_channels * 2, self.base_channels * 4, **c3d_conv_param)
        self.conv3b = ConvModule(self.base_channels * 4, self.base_channels * 4, **c3d_conv_param)
        self.pool3 = nn.AvgPool3d(kernel_size=pool_kernel, stride=pool_stride)

        self.conv4a = ConvModule(self.base_channels * 4, self.base_channels * 8, **c3d_conv_param)
        self.conv4b = ConvModule(self.base_channels * 8, self.base_channels * 8, **c3d_conv_param)

        if self.num_stages == 4:
            self.pool4 = nn.AvgPool3d(kernel_size=pool_kernel, stride=pool_stride)
            self.conv5a = ConvModule(self.base_channels * 8, self.base_channels * 8, **c3d_conv_param)
            self.conv5b = ConvModule(self.base_channels * 8, self.base_channels * 8, **c3d_conv_param)
        
        self.fc1 = nn.Linear(12288, 512)
        self.fc2 = nn.Linear(512, 60)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data. The size of x is (num_batches, 3, 16, 112, 112).

        Returns:
            torch.Tensor: The feature of the input samples extracted by the backbone.
        """
        x = self.conv1a(x)

        x = self.pool1(x)
        x = self.conv2a(x)

        x = self.pool2(x)
        x = self.conv3a(x)
        x = self.conv3b(x)

        x = self.pool3(x)
        x = self.conv4a(x)
        x = self.conv4b(x)

        if self.num_stages == 3:
            return x

        x = self.pool4(x)
        x = self.conv5a(x)
        x = self.conv5b(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.fc2(x)

        x = F.softmax(x, dim=1)

        return x

