import torch
import torch.nn as nn

from ..builder import DECODERS, LOSSES, build_loss
from .attention_decoder import AttentionDecoder1D
from .base import BaseDecoder


@DECODERS.register_module()
class AttentionBidirectionalDecoder(BaseDecoder):
    """Bidirectional Decoder with 1D Attention."""

    def __init__(
        self,
        input_size,
        hidden_size,
        num_classes,
        loss=dict(type="CrossEntropyLoss", ignore_index=0),
        max_decoding_length=35,
    ):
        super(AttentionBidirectionalDecoder, self).__init__()
        self.max_decoding_length = max_decoding_length

        self.forward_decoder = AttentionDecoder1D(
            input_size,
            hidden_size,
            num_classes,
            loss=loss,
            max_decoding_length=max_decoding_length,
        )
        self.backward_decoder = AttentionDecoder1D(
            input_size,
            hidden_size,
            num_classes,
            loss=loss,
            max_decoding_length=max_decoding_length,
        )

    def forward_train(self, x, gt_label, **kwargs):
        """Forward function during training.

        Args:
            x (torch.FloatTensor):  hidden state from encoder. (batch_size x num_steps x num_classes)
            gt_label (torch.LongTensor): the text-index of each image. [batch_size x (max_length+1)]. +1 for [GO] token. text[:, 0] = [GO].
        
        Returns:
            losses (dict): loss dictionary.
        """
        # reverse the hidden features
        x_backward = torch.flip(x, [1])

        # reverse the gt_label
        gt_label_backward = reverse_tensor(gt_label)

        # decode in both forward and backward direction
        loss_l2r = self.forward_decoder.forward_train(x, gt_label)["loss"]
        loss_r2l = self.backward_decoder.forward_train(x_backward, gt_label_backward)[
            "loss"
        ]

        # average two losses
        loss = 0.5 * (loss_l2r + loss_r2l)

        return {"loss": loss, "loss_l2r": loss_l2r, "loss_r2l": loss_r2l}

    def forward_test(self, x, **kwargs):
        """Forward function during inference."""
        # reverse the hidden features
        x_backward = torch.flip(x, [1])

        # decode in both forward and backward direction
        result_l2r = self.forward_decoder.inference(x)
        result_r2l = self.backward_decoder.inference(x_backward)

        # parse result
        probs_l2r = result_l2r["probs"]
        preds_l2r = result_l2r["preds"]
        probs_r2l = result_r2l["probs"]
        preds_r2l = result_r2l["preds"]

        if isinstance(x, torch.Tensor):
            preds = torch.zeros_like(preds_l2r)
        elif isinstance(x, list):
            preds = []
        else:
            raise NotImplementedError("Input should be a torch.Tensor or a list.")

        for i, (pred_l2r, pred_r2l, prob_l2r, prob_r2l) in enumerate(
            zip(preds_l2r, preds_r2l, probs_l2r, probs_r2l)
        ):
            pred = bidirectional_prediction(pred_l2r, pred_r2l, prob_l2r, prob_r2l)
            if isinstance(x, torch.Tensor):
                preds[i, :] = pred
            elif isinstance(x, list):
                preds.append(pred)
            else:
                raise NotImplementedError("Input should be a torch.Tensor or a list.")

        return preds


def reverse_tensor(x, start_token_idx=0, eos_token=1):
    """Reverse gt tensor.
    
    Args:
        x (torch.Tensor): target tensor.
        start_token_idx (int, optional): Index where the start token exists. Defaults to 0.
        eos_token (int, optional): eos token. Defaults to 1.
    
    Returns:
        reversed tensor.
    """
    _, eos_indices = torch.where(x == eos_token)
    x_rev = torch.zeros_like(x)
    for i, ind in enumerate(eos_indices):
        x_rev[i, start_token_idx + 1 : ind] = torch.flip(
            x[i, start_token_idx + 1 : ind], [0]
        )
        x_rev[i, ind] = eos_token
    return x_rev


def bidirectional_prediction(
    pred_forward, pred_backward, prob_forward, prob_backward, eos_token=1,
):
    """Compare product of all probabilities for all steps between forward and backward
        results and return the most likely prediction. Use greedy decoding.

    Args:
        pred_forward (torch.Tensor): Forward prediction.
        pred_backward (torch.Tensor): Backward prediction.
        prob_forward (torch.Tensor): Forward probabilities.
        prob_backward (torch.Tensor): Backward probatilities.
    """
    # parse 'EOS'
    stop_idx = get_stop_idx(pred_forward, eos_token)
    stop_idx_backward = get_stop_idx(pred_backward, eos_token)
    prob_prod_forward = torch.sum(torch.log(prob_forward[:stop_idx]))
    prob_prod_backward = torch.sum(torch.log(prob_backward[:stop_idx_backward]))

    if prob_prod_forward < prob_prod_backward:
        pred_size = int(pred_backward.size(0))
        pred = torch.zeros_like(pred_forward)
        pred[:stop_idx_backward] = pred_backward[:stop_idx_backward].flip(dims=(0,))
        if stop_idx_backward < pred_size:
            pred[stop_idx_backward] = 1
    else:
        pred = pred_forward

    return pred


def get_stop_idx(x, eos_token):
    """Get index where the '[s]' appears in the prediction."""
    stop_idx = torch.where(x == eos_token)[0]
    if len(stop_idx) == 0:
        stop_idx = len(x) - 1
    else:
        stop_idx = int(stop_idx[0])
    return stop_idx
