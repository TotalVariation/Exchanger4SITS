"""
Transformer class.

Copy-paste from torch.nn.Transformer with modifications
borrowed from timm ViT and Mask2Former
"""
import copy
from typing import List, Tuple, Dict, Optional

import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from timm.models.layers import trunc_normal_, DropPath
from layers import MultiHeadAttention, FFN, NormLayer, LayerScale


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                attn_bias: Optional[Tensor] = None,
                **kwargs):

        output = src

        for layer in self.layers:
            output = layer(output,
                           src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask,
                           pos=pos,
                           attn_bias=attn_bias,
                           **kwargs)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        if return_intermediate and norm is not None:
            self.intermediate_norms = _get_clones(norm, num_layers)
            self.norm = None
        else:
            self.intermediate_norms = None

    def forward(self,
                tgt,
                memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,
                memory_pos: Optional[Tensor] = None,
                tgt_attn_bias: Optional[Tensor] = None,
                memory_attn_bias: Optional[Tensor] = None,
                **kwargs):

        output = tgt

        intermediate = []

        for i, layer in enumerate(self.layers):
            output = layer(output,
                           memory,
                           query_mask=tgt_mask,
                           memory_mask=memory_mask,
                           query_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           query_pos=tgt_pos,
                           memory_pos=memory_pos,
                           query_attn_bias=tgt_attn_bias,
                           memory_attn_bias=memory_attn_bias,
                           **kwargs)
            if self.return_intermediate:
                if self.intermediate_norms is not None:
                    intermediate.append(self.intermediate_norms[i](output))
                else:
                    intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        if self.norm is not None:
            output = self.norm(output)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_ratio=4.0,
                 act_type='gelu',
                 norm_type='layernorm',
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 drop=0.,
                 drop_path=0.,
                 init_values=None,
                 **kwargs):

        super().__init__()

        self.self_attn = MultiHeadAttention(dim,
                                            num_heads,
                                            q_proj=True,
                                            k_proj=True,
                                            v_proj=True,
                                            proj_after_attn=True,
                                            qkv_bias=qkv_bias,
                                            qk_scale=qk_scale,
                                            attn_drop=attn_drop,
                                            proj_drop=drop)

        self.ls = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.norm = NormLayer(norm_type, dim)

        self.ffn = FFN(dim,
                       dim_ffd=int(dim * ffn_ratio),
                       num_fcs=2,
                       act_type=act_type,
                       norm_type=norm_type,
                       ffn_drop=drop,
                       drop_path=drop_path)

        self.drop_path = DropPath(drop_prob=drop_path) if drop_path > 0. else nn.Identity()

    def with_pos_embed(self, tensor, pos: Optional[Tensor] = None):
        return tensor if pos is None else tensor + pos

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                attn_bias: Optional[Tensor] = None,
                **kwargs):

        src2 = self.norm(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, v=src2,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask,
                              attn_bias=attn_bias,
                              **kwargs)[0]

        src = src + self.drop_path(self.ls(src2))
        src = self.ffn(src)

        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_ratio=4.0,
                 act_type='gelu',
                 norm_type='layernorm',
                 qkv_bias=True,
                 attn_drop=0.,
                 drop=0.,
                 drop_path=0.,
                 init_values=None,
                 **kwargs):
        super().__init__()

        self.self_attn = MultiHeadAttention(dim,
                                            num_heads,
                                            q_proj=True,
                                            k_proj=True,
                                            v_proj=True,
                                            proj_after_attn=True,
                                            qkv_bias=qkv_bias,
                                            attn_drop=attn_drop,
                                            proj_drop=drop)

        self.cross_attn = MultiHeadAttention(dim,
                                             num_heads,
                                             q_proj=True,
                                             k_proj=True,
                                             v_proj=True,
                                             proj_after_attn=True,
                                             qkv_bias=qkv_bias,
                                             attn_drop=attn_drop,
                                             proj_drop=drop)

        self.norm_query1 = NormLayer(norm_type, dim)
        self.norm_query2 = NormLayer(norm_type, dim)
        self.norm_key = NormLayer(norm_type, dim)

        self.ffn = FFN(dim,
                       dim_ffd=int(dim * ffn_ratio),
                       num_fcs=2,
                       act_type=act_type,
                       norm_type=norm_type,
                       ffn_drop=drop,
                       drop_path=drop_path)

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def with_pos_embed(self, tensor, pos: Optional[Tensor] = None):
        return tensor if pos is None else tensor + pos

    def forward(self,
                query,
                memory,
                query_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                query_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                memory_pos: Optional[Tensor] = None,
                query_attn_bias: Optional[Tensor] = None,
                memory_attn_bias: Optional[Tensor] = None,
                **kwargs):

        q1 = self.norm_query1(query)
        q = k = self.with_pos_embed(q1, query_pos)
        q1 = self.self_attn(q, k, v=q1,
                            attn_mask=query_mask,
                            key_padding_mask=query_key_padding_mask,
                            attn_bias=query_attn_bias,
                            **kwargs)[0]
        query = query + self.drop_path1(self.ls1(q1))
        q2 = self.norm_query2(query)
        memory = self.norm_key(memory)
        q2 = self.cross_attn(q=self.with_pos_embed(q2, query_pos),
                             k=self.with_pos_embed(memory, pos),
                             v=memory,
                             attn_mask=memory_mask,
                             key_padding_mask=memory_key_padding_mask,
                             attn_bias=memory_attn_bias,
                             **kwargs)[0]
        query = query + self.drop_path2(self.ls2(q2))
        query = self.ffn(query)
        return query


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
