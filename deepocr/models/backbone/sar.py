import torch.nn as nn
from mmcv.runner import load_checkpoint

from ...utils import get_root_logger
from ..builder import BACKBONES


class BasicBlock(nn.Module):
    def __init__(self, depth_in, output_dim, kernel_size, stride):
        super(BasicBlock, self).__init__()
        self.identity = nn.Identity()
        self.conv_res = nn.Conv2d(depth_in, output_dim, kernel_size=1, stride=1)
        self.batchnorm_res = nn.BatchNorm2d(output_dim)
        self.conv1 = nn.Conv2d(
            depth_in, output_dim, kernel_size=kernel_size, stride=stride, padding=1
        )
        self.conv2 = nn.Conv2d(
            output_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=1
        )
        self.batchnorm1 = nn.BatchNorm2d(output_dim)
        self.batchnorm2 = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU(inplace=True)
        self.depth_in = depth_in
        self.output_dim = output_dim

    def forward(self, x):
        # create shortcut path
        if self.depth_in == self.output_dim:
            residual = self.identity(x)
        else:
            residual = self.conv_res(x)
            residual = self.batchnorm_res(residual)
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.batchnorm2(out)
        out += residual
        out = self.relu(out)

        return out


@BACKBONES.register_module()
class ResNetSAR(nn.Module):
    """ResNet Backbone for SAR (Show, Attend and Read) model."""

    def __init__(self, in_channels=1, pretrained=None):
        super(ResNetSAR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 1
        self.basicblock1 = BasicBlock(128, 256, kernel_size=3, stride=1)

        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(256)

        # Block 2, 3
        self.basicblock2 = BasicBlock(256, 256, kernel_size=3, stride=1)
        self.basicblock3 = BasicBlock(256, 256, kernel_size=3, stride=1)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(256)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # Block 4,5,6,7,8
        self.basicblock4 = BasicBlock(256, 512, kernel_size=3, stride=1)
        self.basicblock5 = BasicBlock(512, 512, kernel_size=3, stride=1)
        self.basicblock6 = BasicBlock(512, 512, kernel_size=3, stride=1)
        self.basicblock7 = BasicBlock(512, 512, kernel_size=3, stride=1)
        self.basicblock8 = BasicBlock(512, 512, kernel_size=3, stride=1)

        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.batchnorm5 = nn.BatchNorm2d(512)

        # Block 9, 10, 11
        self.basicblock9 = BasicBlock(256, 512, kernel_size=3, stride=1)
        self.basicblock10 = BasicBlock(512, 512, kernel_size=3, stride=1)
        self.basicblock11 = BasicBlock(512, 512, kernel_size=3, stride=1)

        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.batchnorm6 = nn.BatchNorm2d(512)

        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.basicblock1(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.basicblock2(x)
        x = self.basicblock3(x)
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.relu(x)
        x = self.maxpool4(x)
        x = self.basicblock4(x)
        x = self.basicblock5(x)
        x = self.basicblock6(x)
        x = self.basicblock7(x)
        x = self.basicblock8(x)
        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = self.relu(x)
        x = self.basicblock9(x)
        x = self.basicblock10(x)
        x = self.basicblock11(x)
        x = self.conv6(x)
        x = self.batchnorm6(x)
        x = self.relu(x)

        return x
