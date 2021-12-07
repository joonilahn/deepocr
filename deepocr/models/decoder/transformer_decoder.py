import torch
import torch.nn as nn

from ..builder import DECODERS, build_loss
from ..encoder.transformer_encoder import _get_clones, _get_feedforward_net
from .base import BaseDecoder


@DECODERS.register_module()
class TransformerDecoder2D(BaseDecoder):
    """TransformerEncoder is a stack of N encoder layers"""

    def __init__(
        self,
        embed_dim,
        nhead,
        width,
        height,
        num_layers,
        num_classes,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        loss=None,
    ):
        decoder_layer = TransformerDecoderLayer2D(
            embed_dim,
            nhead,
            width,
            height,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        super(TransformerDecoder2D, self).__init__()
        assert loss is not None

        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.num_classes = num_classes + 3
        self.norm = nn.LayerNorm(embed_dim)
        self.last_fc = nn.Linear(embed_dim, num_classes)
        self.criterion = build_loss(loss)

    def loss(self, logits, gt_label):
        """Compute loss for a step."""
        loss = self.criterion(logits.permute(0, 2, 1), gt_label)
        return {"loss": loss}

    def forward_train(
        self,
        tgt,
        memory,
        gt_label,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required). (T, N, E)
            memory: the sequence from the last layer of the encoder (required). (S, N, E)
            tgt_mask: the mask for the tgt sequence (optional). (T, T)
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional). (N, T)
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # exclude 'EOS'
        output = tgt

        for mod in self.layers:
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        output = self.last_fc(output)
        output = output.permute(1, 0, 2).contiguous()  # (b, tgt_len, num_classes)

        losses = self.loss(output, gt_label)

        return losses

    def forward_test(
        self,
        tgt_embed,
        memory,
        tgt_embedding,
        tgt_positional_encoder,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """Forward function during inference."""
        results = self.inference(
            tgt_embed,
            memory,
            tgt_embedding,
            tgt_positional_encoder,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return results["preds"]

    def inference(
        self,
        tgt_embed,
        memory,
        tgt_embedding,
        tgt_positional_encoder,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        tgt_len, batch_size, embed_dim = tgt_embed.size()
        device = tgt_embed.device
        tgt_label = torch.zeros(batch_size, tgt_len).to(torch.int64).to(device)
        output = torch.zeros(batch_size, tgt_len - 1, self.num_classes).to(device)

        for i in range(tgt_len - 1):
            if i > 0:
                # update prediction
                tgt_label[:, i] = cur_preds

                # target embedding
                tgt_embed = tgt_embedding(tgt_label)  # (tgt_len, b, embed_dim)

                # positional encoding for tgt
                tgt_embed = tgt_positional_encoder(tgt_embed)

            output_step = tgt_embed[:-1, :, :].clone()
            for mod in self.layers:
                output_step = mod(
                    output_step,
                    memory,
                    tgt_mask=tgt_mask[:-1, :-1],
                    tgt_key_padding_mask=tgt_key_padding_mask[:, :-1],
                )

            if self.norm:
                output_step = self.norm(output_step)

            output_step = self.last_fc(output_step)[i, :, :]
            cur_preds = torch.argmax(output_step, 1)  # (b, num_classes)

            # update output tensor
            output[:, i, :] = output_step

        probs, preds = output.max(2)

        return {"preds": preds.detach().cpu().data, "probs": probs.detach().cpu().data}


@DECODERS.register_module()
class TransformerDecoderLayer2D(nn.Module):
    r"""TransformerDecoderLayer2D is made up of self-attn and point-wise feedforward network.
    This decoder layer is based on the paper "On Recognizing Texts of Arbitrary Shapes with 2D Self-Attention".
    Junyeop Lee, Sungrae Park, Jeonghun Baek, Seong Joon Oh, Seonghyeon Kim, and Hwalsuk Lee. 2019.

    Args:
        embed_dim: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        # >>> decoder_layer = nn.TransformerDecoderLayer2D(embed_dim=512, nhead=8)
        # >>> memory = torch.rand(200, 32, 512)
        # >>> tgt = torch.rand(35, 32, 512)
        # >>> out = decoder_layer(tgt, memory)
    """

    def __init__(
        self,
        embed_dim,
        nhead,
        width,
        height,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
    ):
        super(TransformerDecoderLayer2D, self).__init__()
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.width = width
        self.height = height

        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.feedforward = _get_feedforward_net(
            embed_dim, "fully-connected", dim_feedforward, activation, dropout=dropout
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        hw, b, embed_dim = memory.size()  # shape: (h*w, b, embed_dim)
        assert hw == self.width * self.height
        assert embed_dim == self.embed_dim

        # Masked Multi-Head Self-Attention
        attn_out = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[
            0
        ]  # shape: (tgt_len, b, embed_dim)

        # Add & Norm
        tgt = tgt + self.dropout1(attn_out)
        tgt = self.norm1(tgt)  # shape: (tgt_len, b, embed_dim)

        # Multi-Head Attention
        # tgt = tgt.permute(1, 0, 2).contiguous()  # shape: (b, tgt_len, embed_dim)
        attn_out = self.multihead_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]

        # Add & Norm
        # tgt = tgt.permute(1, 0, 2).contiguous()  # shape: (tgt_len, b, embed_dim)
        tgt = tgt + self.dropout2(attn_out)
        tgt = self.norm2(tgt)

        # Point-wise Feedforward
        feedforward_out = self.feedforward(tgt)
        # Add & Norm
        tgt = tgt + self.dropout3(feedforward_out)
        tgt = self.norm3(tgt)

        return tgt


@DECODERS.register_module()
class TransformerDecoder1D(nn.Module):
    r"""TransformerDecoder 1D is a stack of N decoder layers
    """

    def __init__(
        self,
        embed_dim,
        nhead,
        num_layers,
        num_classes,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
    ):
        # initialize
        super(TransformerDecoder1D, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            embed_dim, nhead, dim_feedforward, dropout, activation
        )
        decoder_norm = nn.LayerNorm(embed_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers, decoder_norm)

        # last fc layer
        self.num_classes = num_classes + 3
        self.last_fc = nn.Linear(embed_dim, self.num_classes)

        # loss
        self.criterion = build_loss(loss)

    def loss(self, logits, gt_label):
        """Compute loss for a step"""
        loss = self.criterion(logits.permute(0, 2, 1), gt_label)
        return {"loss": loss}

    def forward_train(
        self,
        tgt,
        memory,
        gt_label,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        # exclude 'EOS'
        output = tgt[:-1, :, :]

        output = self.decoder(
            output,
            memory,
            tgt_mask=tgt_mask[:-1, :-1],
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask[:, :-1],
            memory_key_padding_mask=memory_key_padding_mask,
        )

        output = self.last_fc(output)
        output = output.permute(1, 0, 2).contiguous()  # (b, tgt_len, num_classes)

        losses = self.loss(output, gt_label[:, 1:])

        return losses

    def forward_test(
        self,
        tgt_embed,
        memory,
        tgt_embedding,
        tgt_positional_encoder,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """Forward function during inference."""
        results = self.inference(
            tgt_embed,
            memory,
            tgt_embedding,
            tgt_positional_encoder,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return results["preds"]

    def inference(
        self,
        tgt_embed,
        memory,
        tgt_embedding,
        tgt_positional_encoder,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        tgt_len, batch_size, embed_dim = tgt_embed.size()
        device = tgt_embed.device
        tgt_label = torch.zeros(batch_size, tgt_len).to(torch.int64).to(device)
        output = torch.zeros(batch_size, tgt_len - 1, self.num_classes).to(device)
        tgt_embed_step = tgt_embed[:-1, :, :].clone()

        for i in range(tgt_len - 1):
            if i > 0:
                # update prediction
                tgt_label[:, i] = cur_preds

                # target embedding
                tgt_embed = tgt_embedding(tgt_label)  # (tgt_len, b, embed_dim)

                # positional encoding for tgt
                tgt_embed_step = tgt_positional_encoder(tgt_embed_step)[:-1, :, :]

            output_step = tgt_embed_step
            output_step = self.decoder(
                output_step,
                memory,
                tgt_mask=tgt_mask[:-1, :-1],
                memory_mask=None,
                tgt_key_padding_mask=tgt_key_padding_mask[:, :-1],
                memory_key_padding_mask=None,
            )

            output_step = self.last_fc(output_step)[i, :, :]
            cur_preds = torch.argmax(output_step, 1)  # (b, num_classes)

            # update output tensor
            output[:, i, :] = output_step

        probs, preds = output.max(2)

        return {"preds": preds.detach().cpu().data, "probs": probs.detach().cpu().data}
