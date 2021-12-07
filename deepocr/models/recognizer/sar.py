import torch.nn as nn
import torch.nn.functional as F

from ..builder import (RECOGNIZERS, build_backbone, build_decoder,
                       build_encoder, build_pretransform)
from .encoder_decoder import EncoderDecoder


@RECOGNIZERS.register_module()
class SAR(EncoderDecoder):
    """
    Implementation of `Show, Attend, and Read: A Simple and Strong baseline for Irregular
    Text Recognition`<https://arxiv.org/abs/1811.00751>.
    This model has Encoder, Decoder and 2D Attention Layer.
    """

    def __init__(
        self, backbone, neck=None, pretransform=None, encoder=None, decoder=None, pretrained=None,
    ):
        super(SAR, self).__init__(
            backbone,
            neck=neck,
            pretransform=pretransform,
            encoder=encoder,
            decoder=decoder,
            pretrained=pretrained,
        )

    def forward_train(self, img, img_metas, gt_label):
        # pretransform the input image
        if self.pretransform is not None:
            img = self.pretransform(img)

        # extract features
        feature_map = self.extract_feat(img)

        # vertical max pooling
        bsz, hidden_size, h, w = feature_map.size()
        feature_map_pooled = vertical_max_pooling(feature_map, (h, 1)).permute(1, 0, 2)

        # encoding
        holistic_feature = self.encoder(feature_map_pooled)

        # get the last hidden state from the encoder (holistic feature)
        holistic_feature = holistic_feature[-1, ::].unsqueeze(
            0
        )  # shape: [1, bsz, hidden_size]

        # decoding
        result = self.decoder.forward_train(holistic_feature, feature_map, gt_label)

        return result

    def simple_test(self, img, img_metas, **kwargs):
        # pretransform the img image
        if self.pretransform is not None:
            img = self.pretransform(img)

        # extract features
        feature_map = self.extract_feat(img)

        # vertical max pooling
        bsz, hidden_size, h, w = feature_map.size()
        feature_map_pooled = vertical_max_pooling(feature_map, (h, 1)).permute(1, 0, 2)

        # encoding
        holistic_feature = self.encoder(feature_map_pooled)

        # get the last hidden state from the encoder (holistic feature)
        holistic_feature = holistic_feature[-1, ::].unsqueeze(
            0
        )  # shape: [1, bsz, hidden_size]

        # decoding
        result = self.decoder.forward_test(holistic_feature, feature_map)

        return result


def vertical_max_pooling(x, kernel_size):
    return F.max_pool2d(x, kernel_size=kernel_size).squeeze(2).permute(0, 2, 1)
