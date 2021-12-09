import math
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from ..builder import PRETRANSFORMS


def conv3x3_block(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    conv_layer = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)

    block = nn.Sequential(
        conv_layer, nn.BatchNorm2d(out_planes), nn.ReLU(inplace=True),
    )
    return block


class ASTERSTNHead(nn.Module):
    """STN Head for ASTER Transformation."""

    def __init__(
        self, in_channels, num_control_points=20, activation="none", width=288
    ):
        super(ASTERSTNHead, self).__init__()

        self.in_channels = in_channels
        self.num_control_points = num_control_points
        self.activation = activation
        self.stn_convnet = nn.Sequential(
            conv3x3_block(in_channels, 32),  # 32*64
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_block(32, 64),  # 16*32
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_block(64, 128),  # 8*16
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_block(128, 256),  # 4*8
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_block(256, 256),  # 2*4,
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_block(256, 256),
        )  # 1*2

        self.stn_fc1 = nn.Sequential(
            nn.Linear(int(width / (2 ** 5)) * 256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        self.stn_fc2 = nn.Linear(512, num_control_points * 2)

        self.init_weights(self.stn_convnet)
        self.init_weights(self.stn_fc1)
        self.init_stn(self.stn_fc2)

    def init_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()

    def init_stn(self, stn_fc2):
        margin = 0.01
        sampling_num_per_side = int(self.num_control_points / 2)
        ctrl_pts_x = np.linspace(margin, 1.0 - margin, sampling_num_per_side)
        ctrl_pts_y_top = np.ones(sampling_num_per_side) * margin
        ctrl_pts_y_bottom = np.ones(sampling_num_per_side) * (1 - margin)
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        ctrl_points = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0).astype(
            np.float32
        )
        if self.activation == "none":
            pass
        elif self.activation == "sigmoid":
            ctrl_points = -np.log(1.0 / ctrl_points - 1.0)
        stn_fc2.weight.data.zero_()
        stn_fc2.bias.data = torch.Tensor(ctrl_points).view(-1)

    def forward(self, x):
        x = self.stn_convnet(x)
        batch_size, _, h, w = x.size()
        x = x.view(batch_size, -1)
        img_feat = self.stn_fc1(x)
        x = self.stn_fc2(0.1 * img_feat)
        if self.activation == "sigmoid":
            x = F.sigmoid(x)
        x = x.view(-1, self.num_control_points, 2)
        return x
