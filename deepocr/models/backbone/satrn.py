import torch.nn as nn
from mmcv.runner import load_checkpoint

from ...utils import get_root_logger
from ..builder import BACKBONES


@BACKBONES.register_module()
class SATRNBackbone(nn.Module):
    """A basic backbone (shallow CNN) for SATRN model."""

    def __init__(self, in_channels=1, batch_norm=True, hidden_size=512, frozen_stages=-1, pretrained=None):
        super(SATRNBackbone, self).__init__()
        self.frozen_stages = frozen_stages

        if batch_norm:
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(in_channels, hidden_size // 2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(hidden_size // 2),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(hidden_size // 2, hidden_size, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(hidden_size),
                nn.MaxPool2d(2, 2),
            )
        else:
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(in_channels, hidden_size // 2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(hidden_size // 2, hidden_size, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.MaxPool2d(2, 2),
            )
        self.init_weighs(pretrained)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x
