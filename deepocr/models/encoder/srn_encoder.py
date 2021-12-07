import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import ENCODERS, build_encoder, build_posencoder


@ENCODERS.register_module()
class PVAM(nn.Module):
    def __init__(self, input_size, hidden_size, max_decoding_length=35, padding_idx=0):
        super(PVAM, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.max_decoding_length = max_decoding_length + 1
        self.embedding = nn.Embedding(self.max_decoding_length, hidden_size)
        self.e2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

        # word reading order (positional embedding)
        pos_embedding = torch.LongTensor(range(0, self.max_decoding_length))
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, x):
        # input feature map [b x hw x hidden_dim]
        x = self.i2h(x)
        b, hw, e = x.size()

        # reading order vector ([0, 1, ..., N-1])
        pos_embedding = self.embedding(self.pos_embedding)
        pos_embedding = self.e2h(pos_embedding)

        # tile the input feature tensor [b x h*w x e] -> [b x t x h*w x e]
        visual_feature = x.unsqueeze(1).repeat([1, self.max_decoding_length, 1, 1])
        # tile the positional embedding
        pos_embedding = pos_embedding.unsqueeze(1).repeat([1, hw, 1]).unsqueeze(0)

        # get attention map [b x  seq_len x h*w]
        e = self.score(torch.tanh(visual_feature + pos_embedding)).squeeze(-1)
        alpha = F.softmax(e, dim=-1)

        # weighted sum [b x seq_len x hidden_dim]
        aligned_visual_feature = torch.bmm(alpha, x)

        return aligned_visual_feature


@ENCODERS.register_module()
class SRNEncoder(nn.Module):
    def __init__(
        self,
        positional_encoder,
        transformer,
        hidden_size,
        max_decoding_length,
        padding_idx=0,
    ):
        super(SRNEncoder, self).__init__()
        self.positional_encoder = build_posencoder(positional_encoder)
        self.transformer = build_encoder(transformer)
        self.pvam = PVAM(
            hidden_size,
            hidden_size,
            max_decoding_length=max_decoding_length,
            padding_idx=padding_idx
        )

    def forward(self, x):
        # add positional encoding
        x = self.positional_encoder(x)

        # encoder
        x = torch.flatten(x, 2).permute(
            0, 2, 1
        )  # [b, hw, hidden_dim] -> [b, hidden_dim, hw]
        x = self.transformer(x)

        # PVAM
        x = self.pvam(x)

        return x