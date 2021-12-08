"""
VGG-like Backbones
"""

import torch.nn as nn
from mmcv.runner import load_checkpoint

from ...utils import get_root_logger
from ..builder import BACKBONES


@BACKBONES.register_module()
class VGG7(nn.Module):
    """VGG network with 7 convolutional layers."""

    def __init__(
        self, in_channels=1, frozen_stages=-1, batch_norm=True, pretrained=None
    ):
        super(VGG7, self).__init__()

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        self.frozen_stages = frozen_stages
        self.feature_extractor = nn.Sequential()

        def convRelu(i, batch_norm=batch_norm):
            nIn = in_channels if i == 0 else nm[i - 1]
            nOut = nm[i]
            self.feature_extractor.add_module(
                "conv{0}".format(i), nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i])
            )
            if batch_norm:
                self.feature_extractor.add_module(
                    "batchnorm{0}".format(i), nn.BatchNorm2d(nOut)
                )
            self.feature_extractor.add_module("relu{0}".format(i), nn.ReLU(True))

        convRelu(0, batch_norm=False)  # 64 x h x w
        self.feature_extractor.add_module(
            "pooling{0}".format(0), nn.MaxPool2d(2, 2)
        )  # 64 x h/2 x w/2
        convRelu(1, batch_norm=False)  # 128 x h/2 x w/2
        self.feature_extractor.add_module(
            "pooling{0}".format(1), nn.MaxPool2d(2, 2)
        )  # 128 x h/4 x w/4
        convRelu(2, batch_norm=batch_norm)  # 256 x h/4 x w/4
        convRelu(3, batch_norm=False)  # 256 x h/4 x w/4
        self.feature_extractor.add_module(
            "pooling{0}".format(2), nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        )  # 256 x h/8 x w/4
        convRelu(4, batch_norm=batch_norm)  # 512 x h/8 x w/4
        convRelu(5, batch_norm=False)  # 512 x h/8 x w/4
        self.feature_extractor.add_module(
            "pooling{0}".format(3), nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        )  # 512 x h/16 x w/4
        convRelu(6, batch_norm=batch_norm)  # 512 x h/32 (1) x w/4
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages):
                m = getattr(self.feature_extractor, "conv{}".format(i))
                if m is not None:
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False

                # norm layers
                try:
                    m = getattr(self.feature_extractor, "batchnorm{}".format(i))
                    if m is not None:
                        m.eval()
                        for param in m.parameters():
                            param.requires_grad = False
                except:
                    continue

    def forward(self, x):
        # conv features
        features = self.feature_extractor(x)

        return features

    def train(self, mode=True):
        super(VGG7, self).train(mode)
        self._freeze_stages()


@BACKBONES.register_module()
class VGGSAR(nn.Module):
    """Backbone network for 'Show, Attend and Read' model."""

    def __init__(self, in_channels=3, batch_norm=True, frozen_stages=-1, pretrained=None):
        super(VGGSAR, self).__init__()
        self.frozen_stages = frozen_stages

        if batch_norm:
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.MaxPool2d(1, 2),
            )
        else:
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.MaxPool2d(1, 2),
            )
        self.init_weights(pretrained)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
