import torch
import torch.nn as nn

from ..builder import (RECOGNIZERS, build_decoder, build_encoder, build_neck,
                       build_posencoder)
from ..decoder import GSRM, VSFD
from .encoder_decoder import EncoderDecoder


@RECOGNIZERS.register_module()
class SRN(EncoderDecoder):
    def __init__(
        self,
        backbone,
        neck=None,
        pretranform=None,
        encoder=None,
        decoder=None,
        pretrained=None,
    ):
        super(SRN, self).__init__(
            backbone,
            pretranform=pretranform,
            encoder=encoder,
            decoder=decoder,
            pretrained=pretrained
        )

        self.neck = build_neck(neck)
        self.encoder = build_encoder(encoder)
        self.decoder = build_decoder(decoder)

    def forward_train(self, img, img_metas, gt_label):
        # feature extraction
        feature = self.extract_feat(img)

        # encoder
        feature = self.encoder(feature)

        # decoder
        losses = self.decoder.forward_train(feature, gt_label)

        return losses

    def forward_test(self, img, img_metas, **kwargs):
        # feature extraction
        feature = self.extract_feat(img)

        # encoder
        feature = self.encoder(feature)

        # decoder
        result = self.decoder.forward_test(feature)

        return result