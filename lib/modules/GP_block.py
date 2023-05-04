"""
Mostly copy-paste from GPViT
https://github.com/ChenhongyiYang/GPViT
Modified by X.Cai
"""
from typing import Tuple, List, Dict, Optional

import math
import numpy as np

import torch
from torch.nn import functional as F
from torch import nn, Tensor

from timm.models.layers import trunc_normal_, get_act_layer, DropPath

from layers import MultiHeadAttention, MLPMixer, NormLayer, FFN
from .transformer import TransformerEncoderLayer


class LightGroupAttnBlock(nn.Module):
    """
    Lightweight Cross Attention for Updating an External Memory Module.
    """
    def __init__(self,
                 embed_dims,
                 num_heads,
                 norm_type='layernorm',
                 qkv_bias=False,
                 qk_scale=None,
                 proj_drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 **kwargs):
        super(LightGroupAttnBlock, self).__init__()

        self.multihead_attn = MultiHeadAttention(
            embed_dims,
            num_heads,
            q_proj=False,
            k_proj=True,
            v_proj=False,
            proj_after_attn=False,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop
        )

        self.norm_query = NormLayer(norm_type, embed_dims)
        self.norm_key = NormLayer(norm_type, embed_dims)

        self.drop_path = DropPath(drop_prob=drop_path) if drop_path > 0. else nn.Identity()

        self.apply(self._reset_parameters)

    def _reset_parameters(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02, a=-0.02, b=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def with_pos_embed(self, tensor, pos: Optional[Tensor] = None):
        return tensor if pos is None else tensor + pos

    def forward(self,
                query,
                memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                attn_bias: Optional[Tensor] = None,
                return_raw_similarities: bool = False,
                **kwargs):

        q = self.norm_query(query)
        k = self.norm_key(memory)
        attn_out = self.multihead_attn(
            q=self.with_pos_embed(q, query_pos),
            k=self.with_pos_embed(k, memory_pos),
            v=k,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            attn_bias=attn_bias,
            return_raw_similarities=return_raw_similarities,
            **kwargs
        )

        query = query + self.drop_path(attn_out[0])

        if return_raw_similarities:
            return query, *attn_out[1:]
        else:
            return query, attn_out[1]


class FullAttnCatBlock(nn.Module):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 ffn_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 act_type='gelu',
                 norm_type='layernorm',
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 **kwargs):
        super(FullAttnCatBlock, self).__init__()

        self.multihead_attn = MultiHeadAttention(
            embed_dims,
            num_heads,
            q_proj=True,
            k_proj=True,
            v_proj=True,
            proj_after_attn=True,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        self.norm_query = NormLayer(norm_type, embed_dims)
        self.norm_key = NormLayer(norm_type, embed_dims)

        self.ffn = FFN(
            embed_dims,
            dim_ffd=int(embed_dims * ffn_ratio),
            num_fcs=2,
            act_type=act_type,
            norm_type=norm_type,
            ffn_drop=drop,
            drop_path=drop_path,
        )

        self.proj = nn.Linear(embed_dims * 2, embed_dims, bias=True)

        self.drop_path = DropPath(drop_prob=drop_path) if drop_path > 0. else nn.Identity()

        self.apply(self._reset_parameters)

    def _reset_parameters(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02, a=-0.02, b=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def with_pos_embed(self, tensor, pos: Optional[Tensor] = None):
        return tensor if pos is None else tensor + pos

    def forward(self,
                query,
                memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                attn_bias: Optional[Tensor] = None,
                kth_cluster: Optional[int] = None,
                **kwargs):

        q = self.norm_query(query)
        k = self.norm_key(memory)
        q2, attn = self.multihead_attn(
            q=self.with_pos_embed(q, query_pos),
            k=self.with_pos_embed(k, memory_pos),
            v=k,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            attn_bias=attn_bias,
            kth_cluster=kth_cluster,
            **kwargs
        )
        query = torch.cat([query, self.drop_path(q2)], dim=-1)
        query = self.ffn(self.proj(query))

        return query, attn


class GPBlock(nn.Module):
    """
    Feature Update by Exchanging Information (Cross-Attention)
    with an External Memory Module.
    """
    def __init__(self, config, **kwargs):

        super(GPBlock, self).__init__()

        spec_dict = {k.lower(): v for k, v in dict(config.GPBLOCK).items()}
        spec_dict.update(**kwargs) # overwrite
        embed_dims = spec_dict['embed_dims']
        num_group_tokens = spec_dict['num_group_tokens']
        add_pos_token = spec_dict['add_pos_token']
        num_heads = spec_dict['num_heads']
        act_type = spec_dict['act_type']
        norm_type = spec_dict['norm_type']
        ffn_ratio = spec_dict['ffn_ratio']
        qkv_bias = spec_dict['qkv_bias']
        drop = spec_dict['drop']
        attn_drop = spec_dict['attn_drop']
        drop_path = spec_dict['drop_path']
        mixer_depth = spec_dict['mixer_depth']
        mixer_token_expansion = spec_dict['mixer_token_expansion']
        mixer_channel_expansion = spec_dict['mixer_channel_expansion']
        untied_pos_encode = spec_dict['untied_pos_encode']
        pe_dim = spec_dict['pe_dim']

        self.group_token_feat_nwd = nn.Parameter(torch.empty(num_group_tokens, embed_dims))
        trunc_normal_(self.group_token_feat_nwd, std=.02, a=-.02, b=.02)

        if add_pos_token:
            self.group_token_pos_nwd = nn.Parameter(torch.empty(num_group_tokens, embed_dims))
            trunc_normal_(self.group_token_pos_nwd, std=.02, a=-.02, b=.02)

            if untied_pos_encode:
                self.pos_scale = float(embed_dims / num_heads * 2) ** -0.5
                self.pos_q_norm = NormLayer(norm_type, embed_dims)
                self.pos_k_norm = NormLayer(norm_type, pe_dim)
                self.pos_k_proj = nn.Linear(pe_dim, embed_dims)
            else:
                self.pos_proj = nn.Sequential(
                    NormLayer(norm_type, pe_dim),
                    nn.Linear(pe_dim, embed_dims),
                )
        else:
            self.group_token_pos_nwd = None

        _group_attn_cfg = dict(
            embed_dims=embed_dims,
            num_heads=num_heads,
            norm_type=norm_type,
            qkv_bias=qkv_bias,
            qk_scale=self.pos_scale if untied_pos_encode else None,
            proj_drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
        )

        self.group_layer = LightGroupAttnBlock(**_group_attn_cfg)

        _mixer_cfg = dict(
            num_tokens=num_group_tokens,
            embed_dims=embed_dims,
            token_expansion=mixer_token_expansion,
            channel_expansion=mixer_channel_expansion,
            depth=mixer_depth,
            norm_type=norm_type,
            drop_path=drop_path,
            drop_out=drop,
        )

        self.mixer = MLPMixer(**_mixer_cfg)

        _ungroup_attn_cfg = dict(
            embed_dims=embed_dims,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            qk_scale=self.pos_scale if untied_pos_encode else None,
            act_type=act_type,
            norm_type=norm_type,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
        )

        self.ungroup_layer = FullAttnCatBlock(**_ungroup_attn_cfg)

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_group_tokens = num_group_tokens
        self.untied_pos_encode = untied_pos_encode

        self.apply(self._reset_parameters)

    def _reset_parameters(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02, a=-0.02, b=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)

    def get_attn_pos_bias(self, pos_q, pos_k):

        tgt_len, bsz, _ = pos_q.shape
        src_len = pos_k.shape[0]
        pos_q = self.pos_q_norm(pos_q).view(tgt_len, bsz, self.num_heads, -1).permute(1, 2, 0, 3) * self.pos_scale
        pos_k = self.pos_k_proj(self.pos_k_norm(pos_k)).view(
            src_len, bsz, self.num_heads, -1).permute(1, 2, 0, 3)
        attn_pos_bias = torch.einsum('bhic, bhjc -> bhij', pos_q, pos_k)

        return attn_pos_bias

    def forward(
            self,
            x: Tensor,
            pos: Optional[Tensor] = None,
            key_padding_mask: Optional[Tensor] = None,
            kth_cluster: Optional[int] = None,
            **kwargs
    ) -> Tuple[Tensor]:
        """
        Args:
            x: [T, B, C]
            pos: [T, B, C]
            key_padding_mask: [B, T]
        """

        B = x.shape[1]

        # [T, B, C]
        ungroup_tokens = x

        group_token_feat = self.group_token_feat_nwd.unsqueeze(1).expand(-1, B, -1)
        group_token_pos = self.group_token_pos_nwd.unsqueeze(1).expand(-1, B, -1) \
            if self.group_token_pos_nwd is not None else None

        if group_token_pos is not None:
            if self.untied_pos_encode:
                attn_pos_bias = self.get_attn_pos_bias(group_token_pos, pos)
            else:
                pos = self.pos_proj(pos)
        else:
            self.untied_pos_encode = False
            pos = None

        # collect
        group_token_feat = self.group_layer(
            group_token_feat, ungroup_tokens,
            memory_key_padding_mask=key_padding_mask,
            memory_pos=pos if not self.untied_pos_encode else None,
            query_pos=group_token_pos if not self.untied_pos_encode else None,
            attn_bias=attn_pos_bias if self.untied_pos_encode else None,
        )[0]

        # update
        group_token_feat = self.mixer(group_token_feat.transpose(0, 1)).transpose(0, 1)

        # distribute
        ungroup_tokens, attn = self.ungroup_layer(
            ungroup_tokens, group_token_feat,
            memory_mask=key_padding_mask.unsqueeze(-1).expand(
                -1, -1, self.num_group_tokens) if key_padding_mask is not None else None,
            memory_pos=group_token_pos if not self.untied_pos_encode else None,
            query_pos=pos if not self.untied_pos_encode else None,
            attn_bias=attn_pos_bias.transpose(-2, -1) if self.untied_pos_encode else None,
            kth_cluster=kth_cluster,
        )

        return ungroup_tokens, attn.mean(dim=1)
