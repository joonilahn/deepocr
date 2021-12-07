import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import POSENCODERS


@POSENCODERS.register_module()
class AdaptivePositionalEncoder2D(nn.Module):
    """Implemention of adaptive positional encoder from 
    `On Recognizing Texts of Arbitary Shapes with 2D Self-Attention`
    <https://arxiv.org/abs/1910.04396>.
    Adaptive Positional Encoder is a positional encoding layer which is a slightly 
    modified version of positional encoding to be used for 2D visual features."""

    def __init__(self, height, width, embed_dim, dropout=0.1):
        super(AdaptivePositionalEncoder2D, self).__init__()

        # positional encodings for width and height
        positional_encoding_h, positional_encoding_w = make_2D_positional_encoding(
            height, width, embed_dim
        )

        self.register_buffer("positional_encoding_h", positional_encoding_h)
        self.register_buffer("positional_encoding_w", positional_encoding_w)

        # scale factors
        self.dropout = nn.Dropout(dropout)
        self.intermediate_fc = nn.Linear(embed_dim, embed_dim // 2)
        self.scale_factors_fc = nn.Linear(embed_dim // 2, embed_dim * 2)

    def get_scale_factor(self, x):
        b, embed_dim, h, w = x.size()
        inter = F.adaptive_avg_pool2d(x, 1).view(b, embed_dim)
        inter = self.intermediate_fc(inter)
        inter = self.dropout(inter)

        # calculate the scales for the height and the width
        alpha = self.scale_factors_fc(inter)
        alpha = torch.sigmoid(alpha)
        h_scale = alpha[:, :embed_dim]
        w_scale = alpha[:, embed_dim:]

        return h_scale, w_scale

    def forward(self, x):
        b, embed_dim, h, w = x.size()
        h_scale, w_scale = self.get_scale_factor(x)
        h_scale = h_scale.view(b, embed_dim, 1, 1)
        w_scale = w_scale.view(b, embed_dim, 1, 1)

        positional_encoding = (
            h_scale * self.positional_encoding_h + w_scale * self.positional_encoding_w
        )

        return x + positional_encoding


@POSENCODERS.register_module()
class PositionalEncoder(nn.Module):
    """Implementation of positional encoding from 
    `Attention is All You Need`<https://arxiv.org/abs/1706.03762>."""

    def __init__(self, seq_len, embed_dim):
        super(PositionalEncoder, self).__init__()

        self.embed_dim = embed_dim

        # positional encoding
        positional_encoding = make_positional_encoding(seq_len, embed_dim).unsqueeze(
            1
        )  # (seq_len x batch_size x embed_dim)

        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x):
        assert x.size(0) == self.positional_encoding.size(0), (
            "Thr first size of the input tensor should be batch size."
            + "(input size: {}, positional encoding size: {}".format(
                x.size(), self.positional_encoding.size()
            )
        )

        return x + self.positional_encoding


def make_positional_encoding(seq_len, embed_dim):
    # a position tensor along the input sequence
    position_x = torch.arange(0, seq_len, 1).unsqueeze(1)

    # a position tensor along the hidden dimension
    position_dim_even = torch.arange(0, embed_dim, 2)
    position_dim_odd = torch.arange(1, embed_dim + 1, 2)

    # div term
    div_term_even = torch.exp(position_dim_even * (-math.log(10000) / embed_dim))
    div_term_odd = torch.exp(position_dim_odd * (-math.log(10000) / embed_dim))

    # positional encoding
    pos_encoding = torch.zeros(seq_len, embed_dim)
    pos_encoding_even = torch.sin(position_x * div_term_even)
    pos_encoding_odd = torch.cos(position_x * div_term_odd)

    pos_encoding[:, 0::2].copy_(pos_encoding_even)
    pos_encoding[:, 1::2].copy_(pos_encoding_odd)

    return pos_encoding


def make_2D_positional_encoding(h, w, embed_dim):
    # a shape of (1, h, hidden_dim) position tensor along the height
    position_h = make_positional_encoding(h, embed_dim)
    position_h = position_h.permute(1, 0).view(1, embed_dim, h, 1)

    # a shape of (1, w, hidden_dim) position tensor along the width
    position_w = make_positional_encoding(w, embed_dim)
    position_w = position_w.permute(1, 0).view(1, embed_dim, 1, w)

    return position_h, position_w
