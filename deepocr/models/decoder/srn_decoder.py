import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import DECODERS, build_loss
from ..encoder import TransformerEncoder1D
from ..posencoder import PositionalEncoder
from .base import BaseDecoder

class VisualToSemanticBlock(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(VisualToSemanticBlock, self).__init__()
        self.visual_to_semantic = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.visual_to_semantic(x)
        return x


class SemanticReasoningBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_classes,
        num_layers=4,
        nhead=8,
        dim_feedforward=512,
        dropout=0.1,
        max_decoding_length=35,
        padding_idx=0,
    ):
        super(SemanticReasoningBlock, self).__init__()
        self.semantic_to_embedding = nn.Embedding(num_classes, hidden_size)
        self.transformer = TransformerEncoder1D(
            hidden_size,
            nhead,
            num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.fc_reasoning = nn.Lienar(hidden_size, num_classes)
        self.pos_encoder = PositionalEncoder(max_decoding_length + 1, hidden_size)
        memory_mask = self.generate_square_subsequent_mask(max_decoding_length + 1)
        self.register_buffer("memory_mask", memory_mask)

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float('-inf'))
            .masked_fill(mask == 1, float(0))
        )
        return mask

    def forward(self, x):
        # Semantic Reasining Block
        e_prime = self.semantic_to_embedding(x).permute(1, 0, 2)
        e_prime = self.pos_encoder(e_prime)
        gsrm_feature = self.transformer(e_prime, mask=self.memory_mask).permute(1, 0, 2)
        out_gsrm = self.fc_reasoning(gsrm_feature)

        return (gsrm_feature, out_gsrm)


@DECODERS.register_module()
class GSRM(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_classes,
        num_layers=4,
        nhead=8,
        dim_feedforward=512,
        dropout=0.1,
        max_decoding_length=35,
        padding_idx=0,
    ):
        super(GSRM, self).__init__()
        self.visual_to_semantic_block = VisualToSemanticBlock(hidden_size, num_classes)
        self.semantic_to_reasoning_block = SemanticReasoningBlock(
            hidden_size,
            num_classes,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_decoding_length=max_decoding_length,
            padding_idx=padding_idx,
        )
        self.num_classes = num_classes

    def forward(self, x):
        # Visual To Semantic
        out_visual_semantic = self.visual_to_semantic_block(x)

        # prediction based on visual semantic features
        e_prime = torch.argmax(out_visual_semantic, -1)

        # Semantic Reasoning
        gsrm_feature, out_gsrm = self.semantic_to_reasoning_block(e_prime)

        return {
            "visual_semantic_out": out_visual_semantic,
            "gsrm_feature": gsrm_feature,
            "gsrm_out": out_gsrm,
        }


@DECODERS.register_module()
class VSFD(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(VSFD, self).__init__()
        self.fc_fusion = nn.Linear(2 * hidden_size, hidden_size)
        self.fc_pred = nn.Linear(hidden_size, num_classes)

    def forward(self, aligned_visual_feature, gsrm_feature):
        feature_combined = torch.cat([aligned_visual_feature, gsrm_feature], dim=-1)
        feature_combined = self.fc_fusion(feature_combined)
        feature_combined = F.sigmoid(feature_combined)
        feature_fused = (
            feature_combined * aligned_visual_feature
            + (1.0 - feature_combined) * gsrm_feature
        )
        out_vsfd = self.fc_pred(feature_fused)

        return out_vsfd


@DECODERS.register_module()
class SRNDecoder(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        num_classes=None,
        num_layers=4,
        nhead=8,
        dim_feedforward=512,
        dropout=0.1,
        max_decoding_length=35,
        padding_idx=0,
        alpha_e=1.0,
        alpha_r=0.15,
        alpha_f=2.0,
        loss=None
    ):
        super(SRNDecoder, self).__init__()
        assert loss is not None
        self.num_classes = num_classes + 3
        self.gsrm = GSRM(
            hidden_size,
            self.num_classes,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_decoding_length=max_decoding_length,
            padding_idx=padding_idx
        )
        self.vsfd = VSFD(hidden_size, self.num_classes)
        self.criterion = build_loss(loss)
        self.alpha_e = alpha_e
        self.alpha_r = alpha_r
        self.alpha_f = alpha_f

    def loss(self, logits, gt_label):
        """Compute loss for a step."""
        loss = self.criterion(logits.view(-1, logits.shape[-1]), gt_label.view(-1))
        return {"loss": loss}

    def forward_train(self, x, gt_label, **kwargs):
        # GSRM
        gsrm_out_dict = self.gsrm(x)
        visual_semantic_out = gsrm_out_dict["visual_semantic_out"]
        gsrm_feature = gsrm_out_dict["gsrm_feature"]
        gsrm_out = gsrm_out_dict["gsrm_out"]

        # VSFD
        vsfd_out = self.vsfd(x, gsrm_feature)

        # compute losses
        target = gt_label[:, 1:].clone().contiguous()  # exclude 'GO'
        embedding_loss = self.loss(visual_semantic_out, target)["loss"]
        reasoning_loss = self.loss(gsrm_out, target)["loss"]
        decoder_loss = self.loss(vsfd_out, target)["loss"]
        total_loss = (
            self.alpha_e * embedding_loss
            + self.alpha_r * reasoning_loss
            + self.alpha_f * decoder_loss
        )

        return {
            "embedding_loss": embedding_loss,
            "reasoning_loss": reasoning_loss,
            "decoder_loss": decoder_loss,
            "loss": total_loss,
        }

    def forward_test(self, x):
        """Forward function during inference."""
        results = self.inference(x)
        return results["preds"]

    def inference(self, x):
        # GSRM
        gsrm_out_dict = self.gsrm(x)
        gsrm_feature = gsrm_out_dict["gsrm_feature"]

        # VSFD
        vsfd_out = self.vsfd(x, gsrm_feature)
        probs, preds = torch.max(vsfd_out, dim=-1)

        return {"preds": preds, "probs": probs}