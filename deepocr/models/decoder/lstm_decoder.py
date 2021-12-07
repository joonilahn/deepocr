import torch.nn as nn

from ..builder import DECODERS


@DECODERS.register_module()
class LSTMDecoder(nn.Module):
    """Decoder with LSTM network."""

    def __init__(self, input_size, hidden_size, num_classes, bidirectional=True):
        super(LSTMDecoder, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=bidirectional)
        self.embedding = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        hidden, _ = self.rnn(x)
        seq_len, bsz, hidden_size = hidden.size()
        hidden = hidden.view(seq_len * bsz, hidden_size)

        output = self.embedding(hidden)  # [seq_len x batch_size x num_classes]
        output = output.view(seq_len, bsz, -1)

        return output
