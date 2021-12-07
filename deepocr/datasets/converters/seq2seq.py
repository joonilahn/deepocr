import os
from collections.abc import Sequence

import torch

from ..builder import CONVERTERS


@CONVERTERS.register_module()
class Seq2SeqConverter(object):
    """Convert between text-label and text-index for models using cross-entropy loss."""

    def __init__(
        self,
        corpus,
        max_decoding_length=35,
        start_char="[GO]",
        end_char="[s]",
        unk_char="[UNK]",
    ):
        super(Seq2SeqConverter, self).__init__()
        # corpus (str or dict): set of the possible corpus.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        self.corpus_dict = dict()
        self.corpus_dict_inv = dict()
        self.max_decoding_length = max_decoding_length
        self.start_char = start_char
        self.end_char = end_char
        self.unk_char = unk_char

        # 0: start_char, 1: end_char, 2: unk_char, ...
        # for padding, use start_char
        list_token = [
            self.start_char,
            self.end_char,
            self.unk_char,
        ]

        if isinstance(corpus, str):
            if os.path.isfile(corpus):
                with open(corpus, "r", encoding="utf-8") as f:
                    list_characters = f.read().split(",")
            else:
                raise NotImplementedError
        elif isinstance(corpus, (list, tuple)):
            list_characters = list(corpus)
        else:
            raise TypeError

        self.characters = list_token + list_characters

        for i, char in enumerate(self.characters):
            self.corpus_dict[char] = i
            self.corpus_dict_inv[i] = char

    def encode(self, text):
        """Convert text data to indices."""
        if isinstance(text, Sequence) and not isinstance(text, str):
            return self.encode_batch(text)
        elif isinstance(text, str):
            return self.encode_str(text)

    def encode_str(self, text):
        """Encode a single string."""
        # +2 (for [GO] at first step and [s] for the eos). batch_text is padded with [GO] token after [s] token.
        text_encoded = torch.LongTensor(self.max_decoding_length + 2).fill_(0)
        text = list(text)
        text.append(self.end_char)

        # encode text
        iter_num = min(len(text), self.max_decoding_length)
        for i in range(iter_num):
            char_idx = self.char_to_idx(text[i])
            text_encoded[i + 1] = char_idx

        return {"label": text_encoded}

    def encode_batch(self, text):
        """Encode batch data."""
        # +2 (for [GO] at first step and [s] for the eos). batch_text is padded with [GO] token after [s] token.
        batch_text_encoded = torch.LongTensor(
            len(text), self.max_decoding_length + 2
        ).fill_(0)

        # encode text
        for i, t in enumerate(text):
            if len(t) > self.max_decoding_length:
                t = t[: self.max_decoding_length]
            text_encoded = []
            text = list(t)
            text.append(self.end_char)

            for char in text:
                char_idx = self.char_to_idx(char)
                text_encoded.append(char_idx)

            batch_text_encoded[i][1 : 1 + len(text_encoded)] = torch.LongTensor(
                text_encoded
            )  # batch_text_encoded[:, 0] = [GO] token

        return {"label": batch_text_encoded}

    def decode(self, text_index, raw=False, eos_index=1):
        """Convert indices to text data."""
        if isinstance(text_index, list):
            batch_size = len(text_index)
            text_index = text_index[0].expand(size=(1, -1))
        elif isinstance(text_index, torch.Tensor):
            if len(text_index.size()) == 1:
                text_index = text_index.unsqueeze(0)
            batch_size = text_index.shape[0]
        else:
            raise NotImplementedError

        texts = []
        for index in range(batch_size):
            text = []
            for i in text_index[index, :]:
                char = self.idx_to_char(i)
                if (not raw) and (int(i) == eos_index):
                    break
                text.append(char)
            text = "".join(text)
            texts.append(text)

        return texts

    def idx_to_char(self, idx):
        """convert index to character."""
        idx = int(idx)
        return self.corpus_dict_inv[idx]

    def char_to_idx(self, char):
        """convert character to index."""
        if self.corpus_dict.get(char) is not None:
            return self.corpus_dict[char]
        else:
            return self.corpus_dict[self.unk_char]
