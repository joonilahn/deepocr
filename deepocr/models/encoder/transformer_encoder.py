import copy

import torch.nn as nn
from torch.nn.modules.container import ModuleList

from ..builder import ENCODERS


@ENCODERS.register_module()
class TransformerEncoder2D(nn.TransformerEncoder):
    r"""TransformerEncoder is a stack of N encoder layers
    """

    def __init__(
        self,
        embed_dim,
        nhead,
        width,
        height,
        num_layers,
        feedforward="separable",
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
    ):
        encoder_layer = TransformerEncoderLayer2D(
            embed_dim,
            nhead,
            width,
            height,
            feedforward=feedforward,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        norm = nn.LayerNorm(embed_dim)

        # initialize
        super(TransformerEncoder2D, self).__init__(encoder_layer, num_layers, norm=norm)


@ENCODERS.register_module()
class TransformerEncoderLayer2D(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and locality-aware feedforward network.
    This encoder layer is based on the paper "On Recognizing Texts of Arbitrary Shapes with 2D Self-Attention".
    Junyeop Lee, Sungrae Park, Jeonghun Baek, Seong Joon Oh, Seonghyeon Kim, and Hwalsuk Lee. 2019.

    Args:
        embed_dim: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        feedforward: options for the feedforward architecture.
                    separable, fully-connected, and (default="separable")
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(embed_dim=512, nhead=8)
        >>> src = torch.rand(200, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(
        self,
        embed_dim,
        nhead,
        width,
        height,
        feedforward="separable",
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
    ):
        super(TransformerEncoderLayer2D, self).__init__()
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.width = width
        self.height = height

        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.feedforward = _get_feedforward_net(
            embed_dim, feedforward, dim_feedforward, activation, dropout=dropout
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        hw, b, embed_dim = src.size()  # shape: (h, w, b, embed_dim)
        assert (
            hw == self.width * self.height
        ), "First dimension of the input tensor should be width * height"
        assert embed_dim == self.embed_dim

        # Multi-Head Self-Attention
        attn_out = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        # Add & Norm
        attn_out = src + self.dropout1(attn_out)
        attn_out = self.norm1(attn_out)  # shape: (h*w, b, embed_dim)

        # Locality-Aware Feedforward Network
        feedforward_out = self.feedforward(
            attn_out.view(self.height, self.width, b, embed_dim).permute(2, 3, 0, 1)
        )  # shape: (b, embed_dim, h, w)
        feedforward_out = (
            feedforward_out.view(b, embed_dim, hw).permute(2, 0, 1).contiguous()
        )  # shape: (h*w, b, embed_dim)
        # Add & Norm
        feedforward_out = attn_out + feedforward_out
        out = self.norm2(feedforward_out)  # shape: (h*w, b, embed_dim)

        return out


def _get_feedforward_net(
    embed_dim, feedforward, dim_feedforward, activation="relu", dropout=0.1
):
    activation_layer = _get_activation_layer(activation)

    if feedforward == "separable":
        return nn.Sequential(
            nn.Conv2d(embed_dim, dim_feedforward, kernel_size=1),
            activation_layer,
            nn.Dropout(dropout),
            nn.Conv2d(
                dim_feedforward,
                dim_feedforward,
                kernel_size=3,
                padding=1,
                groups=dim_feedforward,
            ),
            activation_layer,
            nn.Dropout(dropout),
            nn.Conv2d(dim_feedforward, embed_dim, kernel_size=1),
            activation_layer,
            nn.Dropout(dropout),
        )
    elif feedforward == "fully-connected":
        return nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            activation_layer,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
            activation_layer,
            nn.Dropout(dropout),
        )
    elif feedforward == "convolution":
        return nn.Sequential(
            nn.Conv2d(embed_dim, dim_feedforward, kernel_size=3, padding=1),
            activation_layer,
            nn.Dropout(dropout),
            nn.Conv2d(dim_feedforward, embed_dim, kernel_size=3, padding=1),
            activation_layer,
            nn.Dropout(dropout),
        )
    else:
        raise RuntimeError(
            "feedforward should be separable, fully-connected or convolution. not %s."
            % feedforward
        )


def _get_activation_layer(activation):
    if activation == "relu":
        return nn.ReLU(True)
    elif activation == "gelu":
        return nn.GELU()
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


@ENCODERS.register_module()
class TransformerEncoder1D(nn.Module):
    r"""TransformerEncoder 1D is a stack of N encoder layers
    """

    def __init__(
        self,
        embed_dim,
        nhead,
        num_layers,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
    ):
        super(TransformerEncoder1D, self).__init__()
        # initialize
        encoder_layer = nn.TransformerDecoderLayer(
            embed_dim, nhead, dim_feedforward, dropout, activation
        )
        encoder_norm = nn.LayerNorm(embed_dim)
        self.encoder = nn.TransformerDecoder(encoder_layer, num_layers, encoder_norm)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        return self.encoder(src, mask=mask, src_key_padding_mask=src_key_padding_mask)