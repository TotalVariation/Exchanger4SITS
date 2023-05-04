from typing import List, Tuple, Dict, Optional

import math
from collections import OrderedDict

import numpy as np

import torch
import torch.nn.functional as F
import torch.distributions as D
from torch import nn, Tensor

from timm.models.layers import trunc_normal_

from ops import XSoftmax


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 kdim=None,
                 vdim=None,
                 q_proj=True,
                 k_proj=True,
                 v_proj=True,
                 proj_after_attn=True,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 **kwargs
                 ):
        super(MultiHeadAttention, self).__init__()

        self.embed_dims = embed_dims
        self.head_dim = embed_dims // num_heads
        assert self.head_dim * num_heads == self.embed_dims, \
            f"embed_dims: {embed_dims} must be divisible by num_heads: {num_heads}"
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.scale_factor = qk_scale or self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dims, embed_dims, bias=qkv_bias) if q_proj else None
        self.k_proj = nn.Linear(kdim if kdim is not None else embed_dims,
                                embed_dims, bias=qkv_bias) if k_proj else None
        self.v_proj = nn.Linear(vdim if vdim is not None else embed_dims,
                                embed_dims, bias=qkv_bias) if v_proj else None

        if proj_after_attn:
            self.proj = nn.Sequential(
                nn.Linear(embed_dims, embed_dims),
                nn.Dropout(proj_drop)
            )
        else:
            self.proj = None

        self.attn_drop = nn.Dropout(attn_drop)

        self.apply(self._reset_parameters)

    def _reset_parameters(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02, a=-0.02, b=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self,
                q: Tensor,
                k: Tensor,
                v: Tensor,
                key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None,
                attn_bias: Optional[Tensor] = None,
                return_raw_similarities: bool = False,
                kth_cluster: Optional[int] = None,
                **kwargs):

        tgt_len, bsz, _ = q.shape
        src_len = k.shape[0]

        q = self.q_proj(q) if self.q_proj is not None else q
        k = self.k_proj(k) if self.k_proj is not None else k
        v = self.v_proj(v) if self.v_proj is not None else v

        # --> [batch, num_heads, seq_len, head_dim]
        q = q.contiguous().view(tgt_len, bsz, self.num_heads, -1).permute(1, 2, 0, 3) * self.scale_factor
        k = k.contiguous().view(src_len, bsz, self.num_heads, -1).permute(1, 2, 0, 3)
        v = v.contiguous().view(src_len, bsz, self.num_heads, -1).permute(1, 2, 0, 3)

        scaled_dot_prod = torch.einsum('b h i d , b h j d -> b h i j', q, k)

        if key_padding_mask is not None:
            # where True indicates elements that will be ignored in the softmax calculation
            assert key_padding_mask.shape == (bsz, src_len), \
                f"expected shape of key_padding_mask is {bsz, src_len}, but got {key_padding_mask.shape}"
            mask = key_padding_mask.view(bsz, 1, 1, src_len).expand(
                -1, self.num_heads, tgt_len, -1).bool()
        else:
            mask = torch.zeros_like(scaled_dot_prod).bool()

        if attn_mask is not None:
            if attn_mask.dim() == 3:
                assert attn_mask.shape == (bsz, tgt_len, src_len), \
                    f'expected shape: {bsz, tgt_len, src_len} but got {attn_mask.shape}'
                attn_mask = attn_mask.view(bsz, 1, tgt_len, src_len)
            elif attn_mask.dim() == 4:
                assert attn_mask.shape == (bsz, self.num_heads, tgt_len, src_len), \
                    f'expected shape: {bsz, self.num_heads, tgt_len, src_len} but got {attn_mask.shape}'
            else:
                raise ValueError(f'attn_mask dims are expected to be 3 or 4 but got {attn_mask.shape}')
            mask = mask.bool() | attn_mask.bool()

        if attn_bias is not None:
            assert attn_bias.shape == scaled_dot_prod.shape, \
                f"expected shape of attn_bias is {bsz, self.num_heads, tgt_len, src_len}, but got {attn_bias.shape}"
            scaled_dot_prod = scaled_dot_prod + attn_bias

        if return_raw_similarities:
            raw_similarities = scaled_dot_prod.clone()

        scaled_dot_prod = (
            scaled_dot_prod -
            scaled_dot_prod.max(dim=-1, keepdim=True).values.detach()).to(q)
        attn = XSoftmax.apply(scaled_dot_prod, mask, -1)
        attn = self.attn_drop(attn)

        if kth_cluster is not None and not self.training:
            attn_values, attn_indices = attn.sort(dim=-1, descending=True)
            attn = torch.zeros_like(attn_values).scatter_(
                -1, attn_indices[..., kth_cluster].unsqueeze(dim=-1), 1.)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.permute(2, 0, 1, 3).contiguous().view(tgt_len, bsz, -1)
        out = self.proj(out) if self.proj is not None else out

        if return_raw_similarities:
            return out, attn, raw_similarities
        else:
            return out, attn

    def extra_repr(self,):
        return f"embed_dims={self.embed_dims}, num_heads={self.num_heads}, head_dim={self.head_dim}, use_bias={self.qkv_bias}"

