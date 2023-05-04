from typing import Optional

import math
from functools import partial

import numpy as np

import torch
from torch.nn import functional as F
from torch import nn, Tensor

from timm.models.layers import trunc_normal_, get_act_layer, DropPath

from .normlayers import NormLayer
from utils import to_2tuple


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class FFN(nn.Module):

    def __init__(self,
                 embed_dims,
                 dim_ffd,
                 num_fcs=2,
                 act_type='gelu',
                 norm_type='layernorm',
                 ffn_drop=0.,
                 drop_path=0.,
                 init_values=None,
                 **kwargs):
        super(FFN, self).__init__()
        # Implementation of Feedforward model
        assert num_fcs >= 2 , f'num_fcs should be no less than 2. got {num_fcs}.'

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, dim_ffd),
                    get_act_layer(act_type)(),
                    nn.Dropout(ffn_drop)
                ))
            in_channels = dim_ffd
        layers.append(nn.Linear(dim_ffd, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)

        self.norm = NormLayer(norm_type, embed_dims)

        self.drop_path = DropPath(drop_prob=drop_path) if drop_path > 0. else nn.Identity()
        self.ls = LayerScale(embed_dims, init_values=init_values) if init_values else nn.Identity()

        self.apply(self._reset_parameters)

    def _reset_parameters(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02, a=-0.02, b=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, tgt: Tensor):
        tgt2 = self.norm(tgt)
        tgt2 = self.layers(tgt2)
        tgt = tgt + self.drop_path(self.ls(tgt2))
        return tgt


class MLP(nn.Module):
    """ Very simple multi-layer perceptron """

    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            num_layers,
            activation='gelu',
    ):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.activation = get_act_layer(activation)()

        self.apply(self._reset_parameters)

    def _reset_parameters(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02, a=-0.02, b=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class GluMlp(nn.Module):
    """ MLP w/ GLU style gating
    See: https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_type='sigmoid',
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % 2 == 0
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        self.chunk_dim = 1 if use_conv else -1

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = get_act_layer(act_type)()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = linear_layer(hidden_features // 2, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        fc1_mid = self.fc1.bias.shape[0] // 2
        nn.init.ones_(self.fc1.bias[fc1_mid:])
        nn.init.normal_(self.fc1.weight[fc1_mid:], std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x, gates = x.chunk(2, dim=self.chunk_dim)
        x = x * self.act(gates)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GatedMlp(nn.Module):
    """ MLP as used in gMLP
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_type='gelu',
            gate_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = get_act_layer(act_type)()
        self.drop1 = nn.Dropout(drop_probs[0])
        if gate_layer is not None:
            assert hidden_features % 2 == 0
            self.gate = gate_layer(hidden_features)
            hidden_features = hidden_features // 2  # FIXME base reduction on gate property?
        else:
            self.gate = nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

        self.apply(self._reset_parameters)
        if gate_layer is not None:
            self.gate.init_weights()

    def _reset_parameters(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02, a=-0.02, b=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.gate(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

