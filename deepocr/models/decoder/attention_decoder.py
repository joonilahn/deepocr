import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import DECODERS, build_loss
from .base import BaseDecoder
from .cell import AttentionCell1D, AttentionCell2D


@DECODERS.register_module()
class AttentionDecoder1D(BaseDecoder):
    """Decoder with 1D Attention."""

    def __init__(
        self, input_size, hidden_size, num_classes, loss=None, max_decoding_length=35,
    ):
        super(AttentionDecoder1D, self).__init__()
        assert loss is not None
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes + 3
        self.attention_cell = AttentionCell1D(
            self.input_size, hidden_size, self.num_classes
        )
        self.generator = nn.Linear(hidden_size, self.num_classes)
        self.max_decoding_length = max_decoding_length
        self.criterion = build_loss(loss)

    def _char_to_onehot(self, input_char, onehot_dim):
        """Convert indices to one-hot tensor."""
        input_char = input_char.detach().cpu().unsqueeze(1)
        batch_size = input_char.size(0)
        zeros = torch.FloatTensor(batch_size, onehot_dim).zero_()
        one_hot = zeros.scatter_(1, input_char, 1)
        return one_hot

    def loss(self, logits, gt_label):
        """Compute loss for a step."""
        loss = self.criterion(logits.view(-1, logits.shape[-1]), gt_label.view(-1))
        return {"loss": loss}

    def forward_train(self, x, gt_label, **kwargs):
        """Forward function during training.

        Args:
            x (torch.FloatTensor):  hidden state from encoder. (batch_size x num_steps x num_classes)
            gt_label (torch.LongTensor): the text-index of each image. [batch_size x (max_length+1)]. +1 for [GO] token. text[:, 0] = [GO].

        Returns:
            losses (dict): loss dictionary.
        """
        batch_size = x.size(0)
        num_steps = self.max_decoding_length + 1  # +1 for [s] at end of sentence.
        device = x.device
        gt_label_no_eos = gt_label[:, :-1].clone()  # exclude 'EOS'

        output_hiddens = torch.zeros(batch_size, num_steps, self.hidden_size).to(device)
        hidden = (
            torch.zeros(batch_size, self.hidden_size).to(device),
            torch.zeros(batch_size, self.hidden_size).to(device),
        )

        # greedy decoding for simple implementation.
        for i in range(num_steps):
            # one-hot vectors for a i-th char. in a batch
            char_onehots = self._char_to_onehot(
                gt_label_no_eos[:, i], self.num_classes
            ).to(device)
            # hidden : decoder's hidden s_{t-1}, x : encoder's hidden H, char_onehots : one-hot(y_{t-1})
            hidden, alpha = self.attention_cell(hidden, x, char_onehots)
            output_hiddens[:, i, :] = hidden[
                0
            ]  # LSTM hidden index (0: hidden, 1: Cell)
        preds = self.generator(output_hiddens)

        target = gt_label[:, 1:].clone().contiguous()  # exclude 'GO'
        losses = self.loss(preds, target)

        return losses

    def forward_test(self, x):
        """Forward function during inference."""
        results = self.inference(x)
        return results["preds"]

    def inference(self, x):
        """Inference."""
        batch_size = x.size(0)
        num_steps = self.max_decoding_length + 1  # +1 for [s] at end of sentence.
        device = x.device

        hidden = (
            torch.zeros(batch_size, self.hidden_size).to(device),
            torch.zeros(batch_size, self.hidden_size).to(device),
        )

        # decoder for inference mode
        targets = torch.LongTensor(batch_size).fill_(0).to(device)  # [GO] token
        logits = torch.zeros(batch_size, num_steps, self.num_classes).to(device)

        for i in range(num_steps):
            char_onehots = self._char_to_onehot(targets, self.num_classes).to(device)
            hidden, alpha = self.attention_cell(hidden, x, char_onehots)
            logits_step = self.generator(hidden[0])
            logits[:, i, :] = logits_step
            _, next_input = logits_step.max(1)
            targets = next_input

        probs, preds = logits.max(2)

        return {"preds": preds.detach().cpu().data, "probs": probs.detach().cpu().data}


@DECODERS.register_module()
class AttentionDecoder2D(nn.Module):
    """Decoder with 2D Attention."""

    def __init__(
        self,
        input_size,
        hidden_size,
        num_classes,
        num_layers=2,
        loss=None,
        max_decoding_length=35,
        padding_idx=0,
    ):
        super(AttentionDecoder2D, self).__init__()
        self.num_classes = num_classes + 3
        self.num_layers = num_layers
        self.decoder = nn.LSTM(input_size, hidden_size, self.num_layers)
        self.attention_cell = AttentionCell2D(input_size, hidden_size)
        self.embedding = nn.Embedding(
            self.num_classes, hidden_size, padding_idx=padding_idx
        )
        self.generator = nn.Linear(2 * hidden_size, self.num_classes)
        self.hidden_size = hidden_size
        self.max_decoding_length = max_decoding_length
        self.criterion = build_loss(loss)

        # initialize weights
        self._reset_parameters()

    def loss(self, logits, gt_label):
        """Compute loss for a step."""
        loss = self.criterion(logits.view(-1, logits.shape[-1]), gt_label.view(-1))
        return {"loss": loss}

    def forward_train(self, holistic_feature, feature_map, gt_label):
        """
        input:
            holistic_feature : holistic feature from the encoder. [1 x batch_size x num_classes]
            feature_map : feature map from the backbone. [batch_size x hidden_size x h//4 x w//8]
            gt_label: ground truth labels. [batch_size x num_steps]
        output: probability distribution at each step [batch_size x num_steps x num_classes]
        """
        device = holistic_feature.device

        # get sizes
        _, batch_size, hidden_size = holistic_feature.size()
        num_steps = self.max_decoding_length + 1
        probs = torch.zeros(batch_size, num_steps, self.num_classes).to(device)

        # decoder for train.py mode
        gt_label_no_eos = gt_label[:, :-1].clone()  # exclude 'EOS'
        tgt_embedding = self.embedding(gt_label_no_eos)
        tgt_embedding = tgt_embedding.permute(
            1, 0, 2
        ).contiguous()  # [num_steps, batch_size, hidden_size]

        # pass the tgt_embedding with the holistic feature to the lstm decoder
        h0 = holistic_feature.repeat((self.num_layers, 1, 1)).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, hidden_size).to(device)
        decoder_hidden, (hn, cn) = self.decoder(tgt_embedding, (h0, c0))

        for i in range(num_steps):
            # hx, cx = self.decoder_cell(tgt_embedding[i], (hx, cx))
            glimpse, alpha = self.attention_cell(decoder_hidden[i], feature_map)
            concat_context = torch.cat(
                [glimpse, decoder_hidden[i]], 1
            )  # batch_size x (hidden_size * 2)
            cur_probs = F.softmax(
                self.generator(concat_context), 1
            )  # batch_size x num_classes
            probs[:, i, :] = cur_probs

        target = gt_label[:, 1:].clone().contiguous()  # exclude 'GO'
        losses = self.loss(torch.log(probs), target)

        return losses

    def forward_test(self, holistic_feature, feature_map):
        """Forward function during inference."""
        results = self.inference(holistic_feature, feature_map)
        return results["preds"]

    def inference(self, holistic_feature, feature_map):
        """Inference method."""
        device = holistic_feature.device

        # get sizes
        _, batch_size, hidden_size = holistic_feature.size()
        num_steps = self.max_decoding_length + 1
        output = torch.zeros(batch_size, num_steps, self.num_classes).to(device)

        for i in range(num_steps):
            if i == 0:
                tgt_embedding = (
                    torch.FloatTensor(1, batch_size, hidden_size).fill_(0.0).to(device)
                )
                hn = holistic_feature.repeat([self.num_layers, 1, 1])
                cn = torch.zeros(self.num_layers, batch_size, hidden_size).to(device)

            else:
                tgt_embedding = self.embedding(cur_preds).view(
                    1, batch_size, hidden_size
                )

            decoder_hidden, (hn, cn) = self.decoder(tgt_embedding, (hn, cn))
            glimpse, alpha = self.attention_cell(decoder_hidden, feature_map)
            concat_context = torch.cat(
                [glimpse, decoder_hidden.view(batch_size, hidden_size)], 1
            )  # batch_size x (hidden_size * 2)
            cur_probs = F.softmax(
                self.generator(concat_context), 1
            )  # batch_size x num_classes
            output[:, i, :] = cur_probs
            _, cur_preds = cur_probs.max(1)

        probs, preds = output.max(2)

        return {"preds": preds.detach().cpu().data, "probs": probs.detach().cpu().data}

    def _reset_parameters(self):
        """Reset parameters."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -1.0, 1.0)
                if m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -1.0, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTMCell):
                for name, param in m.named_parameters():
                    if "bias" in name:
                        nn.init.constant_(param, 0.0)
                    elif "weight" in name:
                        nn.init.xavier_normal_(param)
