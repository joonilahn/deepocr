import torch
import torch.nn as nn

from ..builder import DECODERS, LOSSES, build_loss
from .base import BaseDecoder
from .lstm_decoder import LSTMDecoder


@DECODERS.register_module()
class CTCDecoder(BaseDecoder):
    """Decoder with CTC-Loss."""

    def __init__(
        self,
        input_size,
        hidden_size,
        num_classes,
        blank_idx=0,
        loss=dict(type="CTCLoss", zero_infinity=True),
    ):
        super(CTCDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes + 1  # +1 for unk_char
        self.blank_idx = blank_idx

        self.decoder = nn.Sequential(
            LSTMDecoder(self.input_size, self.hidden_size, self.hidden_size),
            LSTMDecoder(self.hidden_size, self.hidden_size, self.num_classes),
        )

        assert loss is not None
        self.criterion = build_loss(loss)

    def loss(self, logits, gt_label, preds_size, lengths, batch_size):
        """Compute loss for a step."""
        torch.backends.cudnn.enabled = False
        loss = self.criterion(logits, gt_label, preds_size, lengths) / batch_size
        torch.backends.cudnn.enabled = True

        return {"loss": loss}

    def forward_train(self, x, gt_label):
        """Forward function during training.

        Args:
            x (torch.FloatTensor): features from backbone. (batch_size x num_steps x num_classes)
            gt_label (list[list]): the text-index of each image. [batch_size x (max_length+1)]. +1 for [GO] token. text[:, 0] = [GO].
        
        Returns:
            losses (dict): loss dictionary.
        """
        device = x.device
        batch_size = len(gt_label)

        # get lengths
        lengths = torch.sum(gt_label != 0, dim=1).type(torch.IntTensor)
        assert lengths.size(0) == batch_size

        # convert list of gt_label to torch.Tensor
        gt_label = gt_label[gt_label != self.blank_idx].clone()

        # get preds_size for all the batched data
        preds_size = torch.IntTensor([x.size(0)] * batch_size)

        preds_unrolled = self.decoder(x).log_softmax(2)  # (batch_size * seq_len x num_classes)

        # get losses
        losses = self.loss(
            preds_unrolled,
            gt_label.to(device),
            preds_size.to(device),
            lengths.to(device),
            batch_size,
        )

        return losses

    def forward_test(self, x):
        """Forward function during inference."""
        logits = self.decoder(x).log_softmax(2).detach().cpu()
        preds = torch.argmax(logits, 2).permute(1, 0)
        return preds
