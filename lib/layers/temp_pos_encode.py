from typing import Optional
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TemporalPositionalEncoding(nn.Module):

    def __init__(
            self,
            pos_enc_type,
            d_model,
            T=10000.,
            with_gdd_pos=False,
            max_len=None,
            max_val=None,
            scale=None,
            **kwargs
    ):

        super(TemporalPositionalEncoding, self).__init__()

        if max_len is None:
            max_len = 10000 if with_gdd_pos else 500

        scale = 2 * math.pi if scale is None else scale

        if pos_enc_type == 'fourier':
            self.sin_table = None
            self.pe = LearnableFourierPositionalEncoding(
                m=1,
                f=d_model,
                h=32,
                d=d_model)
        elif pos_enc_type == 'rnn':
            sin_table = get_positional_encoding(
                d_model,
                max_len=max_len,
                T=T,
                scale=scale)
            self.sin_table = nn.Embedding.from_pretrained(
                sin_table.float(),
                freeze=True,
                padding_idx=0)
            self.pe = RNNPositionalEncoding(d_model)
        elif pos_enc_type == 'default':
            sin_table = get_positional_encoding(
                d_model,
                max_len=max_len,
                T=T,
                scale=scale)
            self.sin_table = nn.Embedding.from_pretrained(
                sin_table.float(),
                freeze=True,
                padding_idx=0)
        else:
            raise NotImplementedError

        self.max_val = max_len if max_val is None else max_val

        self.pos_enc_type = pos_enc_type
        self.with_gdd_pos = with_gdd_pos

    def forward(
            self,
            temp_idx: Tensor,
            gdd: Optional[Tensor] = None,
            key_padding_mask: Optional[Tensor] = None,
            pos_bias: Optional[Tensor] = None
    ):
        """
        Args:
            temp_idx: LongTensor, shape [batch_size, seq_len]
            gdd: LongTensor, shape [batch_size, seq_len]
            key_padding_mask: BoolTensor, shape [batch_size, seq_len]
                              where True indicates padded positions
        """

        pos = gdd.long() if self.with_gdd_pos else temp_idx.long()
        if key_padding_mask is not None:
            key_padding_mask = ~(key_padding_mask.bool()) # for non-padded elements

        if self.pos_enc_type == 'default':
            pos_encodings = self.sin_table(pos)
        elif self.pos_enc_type == 'rnn':
            pos = self.sin_table(pos)
            pos_encodings = self.pe(pos, key_padding_mask)
        elif self.pos_enc_type == 'fourier':
            pos = pos / self.max_val # normalize to [0, 1]
            pos_encodings = self.pe(pos)
        else:
            raise NotImplementedError

        if pos_bias is not None:
            if key_padding_mask is not None:
                bias = torch.zeros_like(pos_encodings)
                bias[key_padding_mask.nonzero(as_tuple=True)] = pos_bias
            else:
                bias = pos_bias[None, None, :]
            pos_encodings = pos_encodings + bias

        return pos_encodings


def get_positional_encoding(d_model, max_len=500, T=10000, scale=1.):

    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(T) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term * scale)
    pe[:, 1::2] = torch.cos(position * div_term * scale)
    pe = torch.cat([torch.zeros(1, d_model), pe], dim=0)

    return pe


class RNNPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(RNNPositionalEncoding, self).__init__()

        self.rnn = nn.GRU(d_model, d_model, batch_first=True)
        self.mlp = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor, padding_mask: Tensor):

        lengths = padding_mask.float().sum(dim=1)
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                        enforce_sorted=False)
        packed_x, _ = self.rnn(packed_x)
        unpacked_x, _ = pad_packed_sequence(packed_x, batch_first=True)
        out = self.mlp(unpacked_x)

        return out


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, m=1, f=384, h=32, d=768):
        """
        Re-implementation of Learnable Fourier Features
        from https://arxiv.org/abs/2106.02795
        """

        super(LearnableFourierPositionalEncoding, self).__init__()

        assert f % 2 == 0

        self.wr = nn.Linear(m, f//2, bias=False)
        self.mlp = nn.Sequential(
                nn.Linear(f, h),
                nn.GELU(),
                nn.Linear(h, d)
        )
        self.scale = f**-0.5

    def forward(self, x: Tensor):

        x = self.wr(x.unsqueeze(dim=2))
        x = self.scale * torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        x = self.mlp(x)

        return x

