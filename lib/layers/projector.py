"""
Mostly Copy-Paste from https://github.com/naver/trex
"""

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from timm.models.layers import trunc_normal_

from .linear import LinearLayer


class Projector(nn.Module):
    def __init__(
            self,
            feat_dim,
            input_l2_norm=True,
            hidden_layers=3,
            hidden_dim=2048,
            bottleneck_dim=256,
            norm_type='batchnorm',
            act_type='gelu'
    ):
        super().__init__()

        # list of MLP layers
        layers = []

        if input_l2_norm:
            layers.append(L2Norm(dim=1))

        # hidden layers
        _in_dim = feat_dim
        for _ in range(hidden_layers):
            layers.append(
                LinearLayer(
                    _in_dim,
                    hidden_dim,
                    norm_type=norm_type,
                    act_type=act_type,
                    bias=True,
                ))
            _in_dim = hidden_dim

        # bottleneck layer
        layers.append(nn.Linear(_in_dim, bottleneck_dim, bias=False))
        trunc_normal_(layers[-1].weight, std=0.02, a=-0.02, b=0.02)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class ClfLayer(nn.Module):
    """
    Cosine Softmax
    """
    def __init__(self, embed_dim, num_classes, tau=0.1):
        super().__init__()
        self.tau = tau

        self.norm = nn.Identity()
        if tau > 0:
            self.norm = L2Norm(dim=1)

        #self.fc = nn.Linear(embed_dim, num_classes, bias=False)
        self.fc = nn.Parameter(torch.empty(num_classes, embed_dim))
        trunc_normal_(self.fc, std=0.02, a=-0.02, b=0.02)

    def forward(self, x):
        # no temperature scaling
        if self.tau <= 0:
            return torch.matmul(x, self.fc.t())

        # temperature scaling with l2-normalized weights
        x = self.norm(x)
        w = self.norm(self.fc)
        o = torch.matmul(x, w.t()) / self.tau
        return o

    def extra_repr(self):
        return "tau={}".format(self.tau)


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def extra_repr(self):
        return "dim={}".format(self.dim)

    def forward(self, x):
        return F.normalize(x, dim=self.dim, p=2)

