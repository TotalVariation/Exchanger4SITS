from typing import List, Tuple, Dict, Optional

from functools import partial

import torch
from torch.nn import functional as F
from torch import nn, Tensor

from timm.models.layers import trunc_normal_, get_act_layer, DropPath

from .normlayers import NormLayer
from .mlp import GatedMlp


class MLPMixerLayer(nn.Module):
    def __init__(self,
                 num_tokens,
                 embed_dims,
                 token_expansion,
                 channel_expansion,
                 norm_type='layernorm',
                 drop_path=0.,
                 drop_out=0.,
                 **kwargs):

        super(MLPMixerLayer, self).__init__()

        token_mix_dims = int(token_expansion * embed_dims)
        channel_mix_dims = int(channel_expansion * embed_dims)

        self.token_mixer = nn.Sequential(
            nn.Linear(num_tokens, token_mix_dims),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(token_mix_dims, num_tokens),
            nn.Dropout(drop_out)
        )

        self.channel_mixer = nn.Sequential(
            nn.Linear(embed_dims, channel_mix_dims),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(channel_mix_dims, embed_dims),
            nn.Dropout(drop_out)
        )

        self.drop_path1 = DropPath(drop_prob=drop_path)
        self.drop_path2 = DropPath(drop_prob=drop_path)

        self.norm1 = NormLayer(norm_type, embed_dims)
        self.norm2 = NormLayer(norm_type, embed_dims)

    def forward(self, x):
        x = x + self.drop_path1(
            self.token_mixer(self.norm1(x).transpose(1,2)).transpose(1,2))
        x = x + self.drop_path2(self.channel_mixer(self.norm2(x)))
        return x


class MLPMixer(nn.Module):
    def __init__(self,
                 num_tokens,
                 embed_dims,
                 token_expansion=0.5,
                 channel_expansion=4.0,
                 depth=1,
                 norm_type='layernorm',
                 drop_path=0.,
                 drop_out=0.,
                 **kwargs):
        super(MLPMixer, self).__init__()
        layers = [
            MLPMixerLayer(num_tokens, embed_dims, token_expansion, channel_expansion,
                          norm_type, drop_path, drop_out)
            for _ in range(depth)
        ]
        self.layers = nn.Sequential(*layers)

        self.apply(self._reset_parameters)

    def _reset_parameters(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02, a=-0.02, b=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.layers(x)


class SpatialGatingUnit(nn.Module):
    """ Spatial Gating Unit
    Based on: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    def __init__(self, dim, seq_len, norm_type='layernorm'):
        super().__init__()
        gate_dim = dim // 2
        self.norm = NormLayer(norm_type, gate_dim)
        self.proj = nn.Linear(seq_len, seq_len)

    def init_weights(self):
        # special init for the projection gate, called as override by base model init
        nn.init.normal_(self.proj.weight, std=1e-6)
        nn.init.ones_(self.proj.bias)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.proj(v.transpose(-1, -2))
        return u * v.transpose(-1, -2)


class SpatialGatingBlock(nn.Module):
    """ Residual Block w/ Spatial Gating
    Based on: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    def __init__(self, dim, seq_len, mlp_ratio=4, mlp_layer=GatedMlp,
                 norm_type='layernorm', act_type='gelu',
                 drop=0., drop_path=0.):
        super().__init__()
        channel_dim = int(dim * mlp_ratio)
        self.norm = NormLayer(norm_type, dim)
        sgu = partial(SpatialGatingUnit, seq_len=seq_len, norm_type=norm_type)
        self.mlp_channels = mlp_layer(dim, channel_dim, act_type=act_type, gate_layer=sgu, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        return x
