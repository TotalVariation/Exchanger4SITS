"""
Inspired by TSViT https://github.com/michaeltrs/DeepSatModels
Modified by X.Cai
"""
from typing import List, Tuple, Dict, Optional

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from timm.models.layers import trunc_normal_

from ops import XSoftmax
from .transformer import TransformerEncoder, TransformerEncoderLayer
from layers import LinearLayer, NormLayer


class TempSpatTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super(TempSpatTransformer, self).__init__()

        spec_dict = {k.lower(): v for k, v in dict(config.TEMP_SPAT_TRANSFORMER).items()}
        embed_dims = spec_dict['embed_dims']
        temporal_depth = spec_dict['temporal_depth']
        spatial_depth = spec_dict['spatial_depth']
        num_tokens = spec_dict['num_tokens']
        num_heads = spec_dict['num_heads']
        attn_drop = spec_dict['attn_drop']
        drop = spec_dict['drop']
        drop_path = spec_dict['drop_path']
        qkv_bias = spec_dict['qkv_bias']
        ffn_ratio = spec_dict['ffn_ratio']
        act_type = spec_dict['act_type']
        norm_type = spec_dict['norm_type']
        untied_pos_encode = spec_dict['untied_pos_encode']
        use_space_transformer = spec_dict['use_space_transformer']
        in_dim = kwargs['in_channels']

        self.temporal_token_nwd = nn.Parameter(torch.empty(num_tokens, embed_dims))
        trunc_normal_(self.temporal_token_nwd, std=0.02, a=-0.02, b=0.02)
        self.temporal_pos_nwd = nn.Parameter(torch.empty(num_tokens, embed_dims))
        trunc_normal_(self.temporal_pos_nwd, std=0.02, a=-0.02, b=0.02)

        if untied_pos_encode:
            self.pos_q_proj = nn.Linear(embed_dims, embed_dims)
            self.pos_k_proj = nn.Linear(embed_dims, embed_dims)
            self.pos_norm = NormLayer(norm_type, embed_dims)
            self.pos_scale = float(embed_dims / num_heads * 2) ** -0.5

        self.inp_proj = LinearLayer(
            in_dim=in_dim,
            out_dim=embed_dims,
            norm_type='batchnorm',
            act_type='gelu',
            bias=False
        )

        transformer_encoder_layer = TransformerEncoderLayer(
            dim=embed_dims,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            act_type=act_type,
            norm_type=norm_type,
            qkv_bias=qkv_bias,
            qk_scale=self.pos_scale if untied_pos_encode else None,
            attn_drop=attn_drop,
            drop=drop,
            drop_path=drop_path
        )

        self.temporal_transformer = TransformerEncoder(
            transformer_encoder_layer,
            temporal_depth,
            norm=NormLayer(norm_type, embed_dims)
        )

        if use_space_transformer:
            self.space_token_nwd = nn.Parameter(torch.empty(embed_dims))
            trunc_normal_(self.space_token_nwd, std=0.02, a=-0.02, b=0.02)

            transformer_encoder_layer = TransformerEncoderLayer(
                dim=embed_dims,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                act_type=act_type,
                norm_type=norm_type,
                qkv_bias=qkv_bias,
                qk_scale=None,
                attn_drop=attn_drop,
                drop=drop,
                drop_path=drop_path
            )

            self.space_transformer = TransformerEncoder(
                transformer_encoder_layer,
                spatial_depth,
                norm=NormLayer(norm_type, embed_dims)
            )

        self.dropout = nn.Dropout(drop)

        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.untied_pos_encode = untied_pos_encode
        self.use_space_transformer = use_space_transformer
        self.out_dim = embed_dims

        self.apply(self._reset_parameters)

    def _reset_parameters(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02, a=-0.02, b=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)

    def get_attn_pos_bias(self, pos):

        src_len, bsz, _ = pos.shape
        pos = self.pos_norm(pos)
        pos_q = self.pos_q_proj(pos).view(src_len, bsz, self.num_heads, -1).permute(1, 2, 0, 3) * self.pos_scale
        pos_k = self.pos_k_proj(pos).view(src_len, bsz, self.num_heads, -1).permute(1, 2, 0, 3)
        attn_pos_bias = torch.einsum('bhic, bhjc -> bhij', pos_q, pos_k)
        return attn_pos_bias

    def forward(
            self,
            x: Tensor,
            temp_pos: Tensor,
            num_patches: int,
            temporal_mask: Optional[Tensor] = None,
            **kwargs
    ):

        B, T, _ = x.shape
        N = num_patches
        r_bsz = B // N

        x = self.inp_proj(x).transpose(0, 1)

        group_temporal_token = self.temporal_token_nwd.unsqueeze(dim=1).expand(-1, B, -1)
        group_temporal_pos = self.temporal_pos_nwd.unsqueeze(dim=1).expand(-1, B, -1)
        x = torch.cat([group_temporal_token, x], dim=0)
        temp_pos = torch.cat([group_temporal_pos, temp_pos], dim=0)
        temporal_mask = F.pad(temporal_mask, (self.num_tokens, 0), value=0).bool()

        if self.untied_pos_encode:
            attn_pos_bias = self.get_attn_pos_bias(temp_pos)

        x = self.temporal_transformer(
            x,
            src_key_padding_mask=temporal_mask,
            pos=temp_pos if not self.untied_pos_encode else None,
            attn_bias=attn_pos_bias if self.untied_pos_encode else None,
        )
        x = self.dropout(x[:self.num_tokens])

        if self.use_space_transformer:
            x = x.reshape(self.num_tokens, r_bsz, N, -1).permute(2, 1, 0, 3).contiguous().view(N, r_bsz * self.num_tokens, -1)
            cls_space_token = self.space_token_nwd[None, None, :].expand(-1, r_bsz * self.num_tokens, -1)
            x = torch.cat([cls_space_token, x], dim=0)
            x = self.space_transformer(x)[0]
            x = x.reshape(r_bsz, self.num_tokens, -1).mean(dim=1)
        else:
            x = x.reshape(self.num_tokens, r_bsz, N, -1).mean(dim=0).mean(dim=1)

        return x

