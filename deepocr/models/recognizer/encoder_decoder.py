import torch.nn as nn

from ..builder import (RECOGNIZERS, build_backbone, build_decoder,
                       build_encoder, build_neck, build_pretransform)
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class EncoderDecoder(BaseRecognizer):
    """
    Base class for Seq2Seq recognizers.
    
    Seq2Seq recognizers extract features from backbones, then pass the features to
    encoder-decoder modules.
    """

    def __init__(
        self, backbone, neck=None, pretransform=None, encoder=None, decoder=None, pretrained=None,
    ):
        super(EncoderDecoder, self).__init__()
        if pretransform is not None:
            self.pretransform = build_pretransform(pretransform)
        else:
            self.pretransform = None
        if neck:
            self.neck = build_neck(neck)
        else:
            self.neck = None
        self.backbone = build_backbone(backbone)
        self.encoder = build_encoder(encoder)
        self.decoder = build_decoder(decoder)

    def extract_feat(self, x):
        """Directly extract features from the backbone+pretransform."""
        # pretransform the input image
        if self.pretransform is not None:
            x = self.pretransform(x)
        # extract features
        x = self.backbone(x)
        if self.neck:
            x = self.nect(x)
            if isinstance(x, (list, tuple)):
                x = x[0]
        return x

    def forward_train(self, img, img_metas, gt_label):
        # extract features
        features = self.extract_feat(img)

        if len(features.size()) > 3:
            height = features.size(2)
            assert height == 1, "the height of features must be 1"
            features = features.squeeze(2)
            features = features.permute(2, 0, 1)  # [w, b, c]

        # encoder-decoder
        hidden = self.encoder(features)
        hidden = hidden.permute(
            1, 0, 2
        ).contiguous()  # [seq_len, batch_size, embbed_dim]
        losses = self.decoder.forward_train(hidden, gt_label)

        return losses

    def simple_test(self, img, img_metas, **kwargs):
        # extract features
        features = self.extract_feat(img)

        if len(features.size()) > 3:
            height = features.size(2)
            assert height == 1, "the height of features must be 1"
            features = features.squeeze(2)
            features = features.permute(2, 0, 1)  # [w, b, c]

        # encoder-decoder
        hidden = self.encoder(features)
        hidden = hidden.permute(
            1, 0, 2
        ).contiguous()  # [seq_len, batch_size, embbed_dim]
        result = self.decoder.forward_test(hidden)

        return result
