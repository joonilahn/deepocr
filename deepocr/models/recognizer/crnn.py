import torch.nn as nn

from ..builder import (RECOGNIZERS, build_backbone, build_decoder,
                       build_encoder, build_pretransform)
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class CRNN(BaseRecognizer):
    """
    Implementation of CRNN model, `An End-to-End Trainable Neural Network for 
    Image-based Sequence Recognition and Its Application to Scene Text Recognition`,
    Baoguang Shi, Xiang Bai, Cong Yao
    https://arxiv.org/abs/1507.05717
    """

    def __init__(
        self, backbone, pretransform=None, encoder=None, decoder=None, pretrained=None,
    ):
        super(CRNN, self).__init__()
        if pretransform is not None:
            self.pretransform = build_pretransform(pretransform)
        else:
            self.pretransform = None

        self.backbone = build_backbone(backbone)

        if encoder is not None:
            self.encoder = build_encoder(encoder)
        else:
            self.encoder = None

        self.decoder = build_decoder(decoder)

    def extract_feat(self, x):
        """Directly extract features from the backbone+pretransform."""
        # pretransform the input image
        if self.pretransform is not None:
            x = self.pretransform(x)
        # extract features
        x = self.backbone(x)
        return x

    def forward_train(self, img, img_metas, gt_label, **kwargs):
        # extract features
        features = self.extract_feat(img)

        # squeeze the height dimension
        if len(features.size()) > 3:
            height = features.size(2)
            assert height == 1, "the height of features must be 1"
            features = features.squeeze(2)
            features = features.permute(2, 0, 1)  # [w, b, c]

        # encoder-decoder
        if self.encoder is not None:
            features = self.encoder(features)

        losses = self.decoder.forward_train(features, gt_label)

        return losses

    def simple_test(self, img, img_metas, **kwargs):
        # extract features
        features = self.extract_feat(img)

        # squeeze the height dimension
        if len(features.size()) > 3:
            height = features.size(2)
            assert height == 1, "the height of features must be 1"
            features = features.squeeze(2)
            features = features.permute(2, 0, 1)  # [w, b, c]

        # encoder-decoder
        if self.encoder is not None:
            features = self.encoder(features)

        result = self.decoder.forward_test(features)

        return result
