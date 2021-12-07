import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionCell1D(nn.Module):
    """1D Attention Cell."""

    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionCell1D, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(
            hidden_size, hidden_size
        )  # either i2i or h2h should have bias
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(
            torch.tanh(batch_H_proj + prev_hidden_proj)
        )  # batch_size x num_encoder_step * 1

        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(
            1
        )  # batch_size x num_channel
        concat_context = torch.cat(
            [context, char_onehots], 1
        )  # batch_size x (num_channel + num_embedding)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha


class AttentionCell2D(nn.Module):
    """2D Attention Cell."""

    def __init__(self, input_size, hidden_size):
        super(AttentionCell2D, self).__init__()
        self.i2h = nn.Conv2d(input_size, hidden_size, 3, padding=1, bias=False)
        self.h2h = nn.Conv2d(hidden_size, hidden_size)
        self.score = nn.Conv2d(hidden_size, 1, 1, bias=False)
        self.hidden_size = hidden_size

        self._reset_parameters()

    def forward(self, hidden, feature_map):
        """
        input:
            hidden: current hidden state. [batch_size, hidden_size]
            feature_map : feature map from the backbone. [batch_size x hidden_size x h//4 x w//4] 
        output: probability distribution at each step [batch_size x num_steps x num_classes]
        """
        bsz, hidden_size, h, w = feature_map.size()

        # feature map projection
        feature_map_proj = self.i2h(feature_map)

        # hidden_state projection and tile
        hidden_proj = self.h2h(hidden).view(bsz, hidden_size, 1, 1).repeat(1, 1, h, w)

        # fuse the feature map projection and the hidden state projection
        feature_fusion = torch.tanh(feature_map_proj + hidden_proj)

        # linear and softmax
        e = self.score(feature_fusion)  # batch_size x num_encoder_step * 1
        alpha = F.softmax(e.view(bsz, -1), dim=-1).unsqueeze(
            2
        )  # 2D attention map: (bsz, h*w, 1)

        # weighted sum
        glimpse = torch.bmm(feature_map.view(bsz, hidden_size, -1), alpha).squeeze(
            2
        )  # (bsz, hidden_size)

        return glimpse, alpha

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
