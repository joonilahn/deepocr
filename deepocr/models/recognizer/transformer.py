import math

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

from ..builder import RECOGNIZERS, build_posencoder
from .encoder_decoder import EncoderDecoder


@RECOGNIZERS.register_module()
class SATRN(EncoderDecoder):
    """Implementation of SATRN model, `On Recognizing Texts of Arbitrary Shapes
    with 2D Seld-Attention`<https://arxiv.org/abs/1910.04396>"""

    def __init__(
        self,
        backbone,
        neck=None,
        src_positional_encoder=None,
        tgt_positional_encoder=None,
        pretransform=None,
        encoder=None,
        decoder=None,
        embed_dim=512,
        pad_id=0,
        num_classes=None,
        max_decoding_length=35,
        pretrained=None,
    ):
        super(SATRN, self).__init__(
            backbone,
            neck=neck,
            pretransform=pretransform,
            encoder=encoder,
            decoder=decoder,
            pretrained=pretrained,
        )
        self.pad_id = pad_id  # 'GO'
        self.max_decoding_length = max_decoding_length
        self.num_classes = self.decoder.num_classes

        # build positional encoder
        self.src_positional_encoder = build_posencoder(src_positional_encoder)
        tgt_positional_encoder["seq_len"] += 2  # [GO], [EOS]
        self.tgt_positional_encoder = build_posencoder(tgt_positional_encoder)

        # target embedding
        embed_dim = tgt_positional_encoder["embed_dim"]
        self.embedding_tgt = nn.Embedding(
            self.num_classes, embed_dim, padding_idx=pad_id
        )  # +1 for pad_idx ('GO')

        self._reset_parameters()

    def forward_train(self, img, img_metas, gt_label):
        r"""Take in and process masked source/target sequences.

        Args:
            img: the sequence to the encoder.
            img_metas: img metas
            gt_label: the sequence to the decoder.

        Shape:
            - img: :math:`(S, N, E)`.
            - gt_label: :math:`(T, N, E)`.
            - src_mask: :math:`(S, S)`.
            - tgt_mask: :math:`(T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(N, S)`.

            Note: [src/tgt/memory]_mask should be filled with
            float('-inf') for the masked positions and float(0.0) else. These masks
            ensure that predictions for position i depend only on the unmasked positions
            j and are applied identically for each sequence in a batch.
            [src/tgt/memory]_key_padding_mask should be a ByteTensor where True values are positions
            that should be masked with float('-inf') and False values will be unchanged.
            This mask ensures that no information will be taken from position i if
            it is masked, and has a separate mask for each sequence in a batch.

            - output: :math:`(T, N, E)`.

            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decode.

            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number
        """
        # feature extraction
        src = self.extract_feat(img)

        # positional encoding for src
        b, embed_dim, h, w = src.size()
        src = self.src_positional_encoder(src)
        # reshape
        src = (
            src.view(b, embed_dim, h * w).permute(2, 0, 1).contiguous()
        )  # (h*w, b, embed_dim)

        # tgt mask
        device = gt_label.device
        tgt_len = gt_label.size(1)

        # target embedding
        tgt = self.target_embedding(gt_label)  # (tgt_len, b, embed_dim)

        # positional encoding for tgt
        tgt = self.tgt_positional_encoder(tgt)

        # target masks
        tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(device)
        tgt_key_padding_mask = self.generate_key_padding_mask(gt_label).to(device)

        # set other masks
        src_mask = None
        src_key_padding_mask = None
        memory_mask = None
        memory_key_padding_mask = None

        # encoding and decoding
        memory = self.encoder(
            src, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )

        losses = self.decoder.forward_train(
            tgt,
            memory,
            gt_label,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        return losses

    def simple_test(self, img, img_metas, **kwargs):
        # feature extraction
        src = self.extract_feat(img)

        # positional encoding for src
        b, embed_dim, h, w = src.size()
        src = self.src_positional_encoder(src)
        # reshape
        src = (
            src.view(b, embed_dim, h * w).permute(2, 0, 1).contiguous()
        )  # (h*w, b, embed_dim)

        # tgt mask
        device = src.device
        tgt_len = self.max_decoding_length + 2
        tgt_label = torch.zeros(b, tgt_len, device=device).to(torch.int64)

        # target masks
        tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(device)
        tgt_key_padding_mask = self.generate_key_padding_mask(tgt_label, fill_value=None).to(device)

        # encoding and decoding
        memory = self.encoder(src)

        # initial tgt embedding
        tgt_embed = torch.zeros(tgt_len, b, embed_dim).to(device)
        tgt_embed = self.tgt_positional_encoder(tgt_embed)

        result = self.decoder.forward_test(
            tgt_embed,
            memory,
            self.target_embedding,
            self.tgt_positional_encoder,
            tgt_mask=tgt_mask,
            memory_mask=None,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=None,
        )

        return result

    def target_embedding(self, tgt):
        r"""Embed the target tensor.
        """
        tgt = self.embedding_tgt(tgt).permute(1, 0, 2).contiguous()
        tgt *= math.sqrt(tgt.size(2))
        return tgt

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def generate_key_padding_mask(self, text, fill_value=None):
        r"""Generate a key padding mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            The output mask's shape will be the same as that of the input tensor.
        """
        pad_mask = text == self.pad_id
        pad_mask[:, 0] = False

        if fill_value:
            pad_mask.fill_(fill_value)
        return pad_mask

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for name, param in self.named_parameters():
            if name == "encoder" or name == "decoder":
                if param.dim() > 1:
                    xavier_uniform_(param)


@RECOGNIZERS.register_module()
class Transformer1D(SATRN):
    """Transformer 1D mode."""

    def __init__(
        self,
        backbone,
        neck=None,
        src_positional_encoder=None,
        tgt_positional_encoder=None,
        pretransform=None,
        encoder=None,
        decoder=None,
        embed_dim=512,
        pad_id=0,
        num_classes=None,
        max_decoding_length=35,
        pretrained=None,
    ):
        super(Transformer1D, self).__init__(
            backbone,
            neck=neck,
            src_positional_encoder=src_positional_encoder,
            tgt_positional_encoder=tgt_positional_encoder,
            pretransform=pretransform,
            encoder=encoder,
            decoder=decoder,
            embed_dim=embed_dim,
            pad_id=pad_id,
            num_classes=num_classes,
            max_decoding_length=max_decoding_length,
            pretrained=pretrained,
        )

    def forward_train(self, img, img_metas, gt_label):
        # feature extraction
        src = self.extract_feat(img)

        # positional encoding for src
        b, embed_dim, _, len_src = src.size()
        src = src.squeeze(2).permute(2, 0, 1).contiguous()
        src = self.src_positional_encoder(src)

        # tgt mask
        device = gt_label.device
        tgt_len = gt_label.size(1)

        # target embedding
        tgt = self.target_embedding(gt_label)  # (tgt_len, b, embed_dim)

        # positional encoding for tgt
        tgt = self.tgt_positional_encoder(tgt)

        # target masks
        tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(device)
        tgt_key_padding_mask = self.generate_key_padding_mask(gt_label).to(device)

        # set other masks
        src_mask = None
        src_key_padding_mask = None
        memory_mask = None
        memory_key_padding_mask = None

        # encoding and decoding
        memory = self.encoder(
            src, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )

        losses = self.decoder.forward_train(
            tgt,
            memory,
            gt_label,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        return losses

    def simple_test(self, img, img_metas, **kwargs):
        # feature extraction
        src = self.extract_feat(img)

        # positional encoding for src
        b, embed_dim, _, len_src = src.size()
        src = src.squeeze(2).permute(2, 0, 1).contiguous()
        src = self.src_positional_encoder(src)

        # tgt mask
        device = src.device
        tgt_len = self.max_decoding_length + 2
        tgt_label = torch.zeros(b, tgt_len, device=device).to(torch.int64)

        # target masks
        tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(device)
        tgt_key_padding_mask = self.generate_key_padding_mask(tgt_label, fill_value=None).to(device)

        # encoding and decoding
        memory = self.encoder(src)

        # initial tgt embedding
        tgt_embed = torch.zeros(tgt_len, b, embed_dim).to(device)
        tgt_embed = self.tgt_positional_encoder(tgt_embed)

        result = self.decoder.forward_test(
            tgt_embed,
            memory,
            self.target_embedding,
            self.tgt_positional_encoder,
            tgt_mask=tgt_mask,
            memory_mask=None,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=None,
        )

        return result
