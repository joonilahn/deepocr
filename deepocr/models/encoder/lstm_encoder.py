import torch.nn as nn

from ..builder import ENCODERS


@ENCODERS.register_module()
class LSTMEncoder(nn.Module):
    """Encodeㄱ with LSTM Network."""

    def __init__(
        self, input_size, hidden_size, num_layers, cell_type="LSTM", bidirectional=True
    ):
        super(LSTMEncoder, self).__init__()
        if cell_type == "LSTM":
            self.encoder = nn.LSTM(
                input_size, hidden_size, num_layers, bidirectional=bidirectional
            )
        elif cell_type == "GRU":
            self.encoder = nn.GRU(input_size, hidden_size, bidirectional=bidirectional)
        else:
            raise NotImplementedError(
                "{} is not a valid type for the encoder.".format(cell_type)
            )

        if bidirectional:
            self.embedding = nn.Linear(hidden_size * 2, hidden_size)
        else:
            self.embedding = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        recurrent, _ = self.encoder(x)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, hidden_size]
        output = output.view(T, b, -1)

        return output
