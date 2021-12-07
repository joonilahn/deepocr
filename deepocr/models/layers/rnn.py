import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    """Sequential model of Bidirectional LSTM network and Linear network."""

    def __init__(self, input_size, hidden_size, num_layers=1):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        self.embedding = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, hidden_size]
        output = output.view(T, b, -1)

        return output


class BidirectionalGRU(nn.Module):
    """Sequential model of Bidirectional GRU network and Linear network."""

    def __init__(self, input_size, hidden_size):
        super(BidirectionalGRU, self).__init__()

        self.rnn = nn.GRU(input_size, hidden_size, bidirectional=True)
        self.embedding = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, hidden_size]
        output = output.view(T, b, -1)

        return output
