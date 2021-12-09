import os
from collections.abc import Sequence

import numpy as np
import torch

from ..builder import CONVERTERS


@CONVERTERS.register_module()
class CTCConverter(object):
    """Convert between text-label and text-index for models using ctc loss."""

    def __init__(
        self, corpus, max_decoding_length=35, unk_char="[UNK]", blank_char="|",
    ):
        """
        Args:
            corpus (str): a csv file path containing corpus data.
            max_decoding_length (int, optional):
                        maximum number of characters in a single data.
                        Defaults to 35.
            unk_char (str, optional): a key for unknown char. Defaults to "[UNK]".
            blank_char (str, optional): a key for blank char. Defaults to "|".
        
        Returns:
            [type]: [description]
        """
        super(CTCConverter, self).__init__()
        self.corpus_dict = dict()
        self.corpus_dict_inv = dict()
        self.max_decoding_length = max_decoding_length
        self.unk_char = unk_char
        self.blank_char = blank_char

        if os.path.isfile(corpus):
            with open(corpus, "r", encoding="utf-8") as f:
                list_characters = f.read().replace(",", "")
        else:
            raise ValueError

        # 0 index for blank_char, -1 index for extra character
        self.corpus_dict[self.blank_char] = 0
        self.corpus_dict_inv[0] = self.blank_char
        self.corpus_dict[self.unk_char] = len(list_characters) + 1
        self.corpus_dict_inv[len(list_characters) + 1] = self.unk_char

        for i, char in enumerate(list_characters):
            self.corpus_dict[char] = i + 1
            self.corpus_dict_inv[i + 1] = char

    def encode(self, text):
        """Convert text data to indices."""
        if isinstance(text, Sequence) and not isinstance(text, str):
            return self.encode_batch(text)
        elif isinstance(text, str):
            return self.encode_str(text)

    def encode_str(self, text):
        """Encode a single string."""
        # +1 (for [GO] at first step and [s] for the eos). batch_text is padded with [GO] token after [s] token.
        text_encoded = torch.IntTensor(self.max_decoding_length).fill_(0)

        # encode text
        iter_num = min(len(text), self.max_decoding_length)
        for i in range(iter_num):
            char_idx = self.char_to_idx(text[i])
            text_encoded[i] = char_idx

        return {"label": text_encoded}

    def encode_batch(self, text):
        """Encode batch data."""
        # +2 (for [GO] at first step and [s] for the eos). batch_text is padded with [GO] token after [s] token.
        batch_text_encoded = torch.IntTensor(len(text), self.max_decoding_length).fill_(
            0
        )

        # encode text
        for i, t in enumerate(text):
            if len(t) > self.max_decoding_length:
                t = t[: self.max_decoding_length]
            text_encoded = []

            for char in t:
                char_idx = self.char_to_idx(char)
                text_encoded.append(char_idx)

            batch_text_encoded[i][: len(text_encoded)] = torch.IntTensor(text_encoded)

        return {"label": batch_text_encoded}

    def decode(self, texts_encoded, gt=False, raw=False):
        """Convert indices to text data."""
        texts_decoded = []
        if gt:
            decode_method = self.decode_single_gt
        else:
            decode_method = self.decode_single_pred

        if isinstance(texts_encoded, torch.Tensor) or isinstance(
            texts_encoded, np.ndarray
        ):
            if len(texts_encoded.shape) > 1:
                assert len(texts_encoded.shape) == 2
                batch_size = texts_encoded.shape[0]

                for i in range(batch_size):
                    text_encoded = texts_encoded[i]
                    texts_decoded.append(decode_method(text_encoded, raw=raw))

            elif len(texts_encoded.shape) == 1:
                texts_decoded = decode_method(texts_encoded, raw=raw)

        elif isinstance(texts_encoded, list):
            assert isinstance(texts_encoded[0], int)
            texts_decoded = decode_method(texts_encoded, raw=raw)

        return texts_decoded

    def decode_single_pred(self, text_encoded, raw=False):
        """Greedy decoding a single prediction data."""
        if raw:
            return "".join([self.idx_to_char(i) for i in text_encoded])
        else:
            text_decoded = ""
            blank_index = self.char_to_idx(self.blank_char)
            char_prev = None
            for i, char_cur in enumerate(text_encoded):
                if char_cur != blank_index and (not (i > 0 and char_prev == char_cur)):
                    text_decoded += self.idx_to_char(char_cur)
                char_prev = char_cur
        return "".join(text_decoded)

    def decode_single_gt(self, text_encoded, raw=False):
        """Decode a single gt data."""
        text_encoded = text_encoded[text_encoded != 0]
        return "".join([self.idx_to_char(i) for i in text_encoded])

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
