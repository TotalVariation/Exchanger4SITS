"""
Mostly Copy-Paste from
https://github.com/VSainteuf/lightweight-temporal-attention-pytorch
Modified by X.Cai
"""

from typing import Tuple, Optional

import math
import copy

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from timm.models.layers import trunc_normal_, get_act_layer

from layers import LinearLayer, TemporalPositionalEncoding
from ops import XSoftmax


class LTAE(nn.Module):
    def __init__(self, config, **kwargs):
        super(LTAE, self).__init__()

        spec_dict = {k.lower(): v for k, v in dict(config.LTAE).items()}
        mlp_dims = spec_dict['mlp_dims']
        with_pos_enc = spec_dict['with_pos_enc']
        pos_enc_type = spec_dict['pos_enc_type']
        with_gdd_pos = spec_dict['with_gdd_pos']
        pe_t = spec_dict['pe_t']
        max_temp_len = spec_dict['max_temp_len']
        d_model = spec_dict['d_model']
        d_k = spec_dict['d_k']
        n_head = spec_dict['n_head']
        norm_type = spec_dict['norm_type']
        act_type = spec_dict['act_type']
        dropout = spec_dict['dropout']

        in_channels = kwargs['in_channels']
        if in_channels != d_model:
            self.in_layer = LinearLayer(
                in_channels,
                d_model,
                norm_type=norm_type,
                act_type=act_type,
                bias=False,
            )
        else:
            self.in_layer = None

        if with_pos_enc:
            self.pe = TemporalPositionalEncoding(
                pos_enc_type,
                d_model//n_head,
                with_gdd_pos=with_gdd_pos,
                T=pe_t,
                max_len=max_temp_len,
            )
        else:
            self.pe = None

        self.decoder_head = LightTemporalAttention(
            d_model,
            d_k,
            n_head,
            dropout=dropout
        )

        layers = []
        for i in range(len(mlp_dims)):
            layers.append(LinearLayer(
                d_model if i == 0 else mlp_dims[i-1],
                mlp_dims[i],
                norm_type=norm_type,
                act_type=act_type,
                bias=False,
            ))
        self.mlp = nn.Sequential(*layers)

        self.dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.last_dim = mlp_dims[-1]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02, a=-0.02, b=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)

    def forward(
            self,
            x: Tensor,
            temp_idx: Tensor,
            gdd: Optional[Tensor] = None,
            key_padding_mask: Optional[Tensor] = None,
            **kwargs
    ) -> Tuple[Tensor]:

        """
        Shape:
            -x: [B, T, C]
            -temp_idx: [B, T]
            -gdd: [B, T]
            -key_padding_mask: [B, T] where True indicates padded positions.
        """

        if self.in_layer is not None:
            x = self.in_layer(x, key_padding_mask)

        if self.pe is not None:
            pos = self.pe(temp_idx, gdd, key_padding_mask)
            x = x + pos.repeat(1, 1, self.n_head)

        out, attn = self.decoder_head(
            x,
            key_padding_mask=key_padding_mask,
            **kwargs)

        out = self.dropout(self.mlp(out))

        return out, attn


class LightTemporalAttention(nn.Module):

    def __init__(self, d_model, d_k, n_head, dropout=0.):
        super(LightTemporalAttention, self).__init__()

        self.to_k = nn.Linear(d_model, n_head * d_k, bias=True)
        self.query = nn.Parameter(torch.zeros(n_head, d_k))
        nn.init.normal_(self.query, mean=0, std=np.sqrt(2.0 / d_k))

        self.scale_factor = 1.0 / np.sqrt(d_k)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_k

    def forward(
            self,
            x: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            **kwargs
    ) -> Tuple[Tensor]:

        B, T, C = x.size()

        q = self.query.expand(B, -1, -1) * self.scale_factor # (B, nh, d_k)
        k = self.to_k(x).view(B, T, self.n_head, self.d_k).permute(0, 2, 1, 3)  # (B, nh, T, d_k)
        v = x.view(B, T, self.n_head, C//self.n_head).permute(0, 2, 1, 3) # (B, nh, T, d_v)

        scaled_dot_prod = torch.einsum('b h d, b h j d -> b h j', q, k)
        scaled_dot_prod = (scaled_dot_prod - scaled_dot_prod.max(dim=-1, keepdim=True).values.detach()).to(q)

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (B, T), \
                f"expected shape of key_padding_mask is {B, T}, but got {key_padding_mask.shape}"
            mask = key_padding_mask.unsqueeze(dim=1).expand(-1, self.n_head, -1)
        else:
            mask = torch.zeros_like(scaled_dot_prod)

        attn = XSoftmax.apply(scaled_dot_prod, mask, -1)
        attn = self.dropout(attn)

        out = torch.einsum('b h j, b h j d -> b h d', attn, v)
        out = out.view(B, -1)

        return out, attn
