from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch.nn as nn


class BaseDecoder(nn.Module, metaclass=ABCMeta):
    """Base class for recognizers."""

    def __init__(self):
        super(BaseDecoder, self).__init__()

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        pass

    @abstractmethod
    def forward_train(self, hidden, gt_label, **kwargs):
        """Forward function during training."""
        pass

    def forward_test(self, hidden, **kwargs):
        """Test without augmentation."""
        pass

    def forward(self, hidden, gt_label, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(hidden, gt_label, **kwargs)
        else:
            return self.forward_test(hidden, **kwargs)
